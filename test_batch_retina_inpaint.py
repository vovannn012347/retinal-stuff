import argparse
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as t_tf_fn
import utils.misc_inpaint as misc
from utils.retinaldata import get_image_bbox, is_image_file


def get_image_with_name_any_extension(folder_path, image_name):
    search_pattern = os.path.join(folder_path, f"{image_name}.*")

    image_files = glob.glob(search_pattern)

    # Check if any image file is found
    if image_files:
        return image_files[0]  # Return the first found image (if multiple, modify as needed)
    else:
        return None


parser = argparse.ArgumentParser(description='Test inpainting')

parser.add_argument("--checkpoint", type=str,
                    default="training-data/retina-infill/checkpoints/states_gen_test.pth",
                    help="path to the checkpoint file")


parser.add_argument("--image-dir", type=str,
                    default="training-data/retina-detection/training-images-batch-processing-output",
                    help="path to the image source folder")

parser.add_argument("--mask-dir", type=str,
                    default="training-data/retina-detection/training-masks-batch-processing-output",
                    help="path to the cropped image output folder")

parser.add_argument("--image-dir-out", type=str,
                    default="training-data/retina-infill/batch-output",
                    help="path to the cropped images mask output folder")

img_shapes = [576, 576, 3]

def main():

    args = parser.parse_args()

    generator_state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks_inpaint import Generator
    else:
        from model.networks_inpaint_tf import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')

    channels = 3
    # set up network
    generator = Generator(cnum_in=channels+2, cnum_out=channels, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['G']
    generator.load_state_dict(generator_state_dict, strict=True)

    for sample in [entry for entry in os.scandir(args.image_dir) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        # load image and mask
        name, extension = os.path.splitext(sample.name)
        if extension.endswith("jpg"):
            extension = ".png"

        mask_name = get_image_with_name_any_extension(args.mask_dir, name)
        if mask_name is None:
            print(f"Mask not found, skipping")
            continue

        image = Image.open(sample.path).convert('RGB')
        img_bbox = get_image_bbox(image)
        image = image.crop(img_bbox)

        mask = Image.open(mask_name).convert('L')
        mask = mask.crop(img_bbox)

        # prepare input
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        image = T.Resize(img_shapes[:2], antialias=True)(image)
        mask = T.Resize(img_shapes[:2], antialias=True)(mask)

        # _, h, w = image.shape
        # grid = 8

        # print(f"Shape of image: {image.shape}")
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        # mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        # print(f"Shape of image: {image.shape}")

        image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
        # target_color = 0
        # tolerance = 0.23
        # mask = magic_wand_mask_selection(mask).to(torch.float32)
        mask = misc.dilate_mask(mask, 3)

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        zeroes_x = torch.zeros_like(image_masked)
        x = torch.cat([image_masked, ones_x, ones_x*mask], dim=1)  # concatenate channel

        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        # complete image
        image_inpainted = image * (1.-mask) + x_stage2 * mask

        img_out = misc.pt_to_image(image_inpainted).squeeze().cpu()
        img_out = t_tf_fn.to_pil_image(img_out, mode="RGB")

        img_name_path = os.path.join(args.image_dir_out, name + extension)
        img_out.save(img_name_path)

        print(f"Saved output file at: {img_name_path}")


if __name__ == '__main__':
    main()
