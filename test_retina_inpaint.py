import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as t_tf_fn
import utils.misc_inpaint as misc
from utils.retinaldata import get_image_bbox

parser = argparse.ArgumentParser(description='Test inpainting')
parser.add_argument("--checkpoint", type=str,
                    default="training-data/retina-infill/checkpoints/states_gen_test.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image", type=str,
                    default="training-data/retina-infill/training-images/001АЄЯ01.jpg",
                    help="path to the image file")
parser.add_argument("--mask", type=str,
                    default="training-data/retina-infill/training-masks/001АЄЯ01.mask.jpg",
                    help="path to the mask file")
parser.add_argument("--out", type=str,
                    default="training-data/retina-infill/output/001АЄЯ01_out.png",
                    help="path for the output file")


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

    # load image and mask
    image = Image.open(args.image).convert('RGB')
    img_bbox = get_image_bbox(image)
    image = image.crop(img_bbox)
    mask = Image.open(args.mask).convert('L')
    mask = mask.crop(img_bbox)

    # prepare input
    img_shapes = [576, 576, 3]
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)
    image = T.Resize(img_shapes[:2], antialias=True)(image)
    mask = T.Resize(img_shapes[:2], antialias=True)(mask)

    # _, h, w = image.shape
    # grid = 8

    print(f"Shape of image: {image.shape}")
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    # image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    # mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    # target_color = 0
    # tolerance = 0.23
    # misc.dilate_mask(((abs(mask - target_color) > tolerance).to(dtype=torch.float32, device=device)), 3)
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
    img_out.save(args.out)

    # save inpainted image
    # img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    # img_out = img_out.to(device='cpu', dtype=torch.uint8)
    # img_out = Image.fromarray(img_out.numpy())
    # img_out.save(args.out)

    print(f"Saved output file at: {args.out}")


if __name__ == '__main__':
    main()
