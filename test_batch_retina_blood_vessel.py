import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from model.retina_network import RetinalConvolutionNetwork
from utils.misc_retina import gaussian_2d, magic_wand_mask_selection_faster
from utils.retinaldata import get_image_bbox, is_image_file

parser = argparse.ArgumentParser(description='Test retina detection in batch')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-detection/checkpoints/states_2024_09_05.pth",
                    help="path to the checkpoint file")

'''d:/Learning/Аспірантура/Аспірантська/kaggle_retina_dataset/preprocessed_images'''
'''training-data/retina-detection/training-images'''
parser.add_argument("--image-dir", type=str,
                    default="d:/Learning/Аспірантура/Аспірантська/kaggle_retina_dataset/preprocessed_images",
                    help="path to the image source folder")

parser.add_argument("--image-dir-out", type=str,
                    default="training-data/retina-detection/training-images-batch-processing-output",
                    help="path to the cropped image output folder")

parser.add_argument("--image-dir-mask-out", type=str,
                    default="training-data/retina-detection/training-masks-batch-processing-output",
                    help="path to the cropped images mask output folder")


image_patch_stride = [32, 32]
#image_target_size_multiple = [64, 64] #image_patch_stride * 4
image_patch_size = [96, 96] #int(image_patch_stride * 4)
image_target_size = [576, 576] #image_target_size_multiple * 10
input_size_in = 3

def split_by_position(string, position):
    return [string[:position], string[position:]]

def main():
    args = parser.parse_args()

    # set up network
    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    convolution_pass1 = RetinalConvolutionNetwork(cnum_in=input_size_in, cnum_out=1).to(device)  # cnum in is 1 channel
    convolution_pass1.load_state_dict(convolution_state_dict['convolution_pass1'], strict=True)

    convolution_pass2 = RetinalConvolutionNetwork(cnum_in=input_size_in + 1, cnum_out=1).to(device)
    convolution_pass2.load_state_dict(convolution_state_dict['convolution_pass2'], strict=True)

    gauss_patch = gaussian_2d([image_patch_size[0], image_patch_size[1]],
                              min_value=0.5, max_value=1.6, sigma=20)
    gauss_count = None
    to_pil_transform = T.ToPILImage(mode='L')

    for sample in [entry for entry in os.scandir(args.image_dir) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")

        #old code

        pil_image = Image.open(sample.path).convert('RGB')  # 3 channel
        img_bbox = get_image_bbox(pil_image)

        pil_image = pil_image.crop(img_bbox)
        pil_image = pil_image.resize((image_target_size[0], image_target_size[1]), Image.Resampling.BICUBIC)

        image_pass = T.ToTensor()(pil_image)
        image_pass = TF.adjust_sharpness(image_pass, 2)

        # image result output
        img_out = TF.to_pil_image(image_pass.cpu(), mode="RGB")
        name, extension = os.path.splitext(sample.name)
        if extension.endswith("jpg"):
            extension = ".png"

        img_out.save(os.path.join(args.image_dir_out, name + extension))

        channels, h, w = image_pass.shape
        # image_temp_mask = torch.zeros([1, h, w])
        # image_temp_mask = image_temp_mask * 2 - 1 #[0, 1] -> [-1, 1]
        image_pass.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]

        image_patches_unfolded_pass1 = image_pass.unfold(1, image_patch_size[0], image_patch_stride[0])
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.unfold(2, image_patch_size[1],
                                                                           image_patch_stride[1])
        _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass1.size()
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.contiguous()
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.view(channels, -1,
                                                                         image_patch_size[0], image_patch_size[1])
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.permute(1, 0, 2, 3)

        mask_pass1 = convolution_pass1(image_patches_unfolded_pass1)  # outputs sigmoid [0.0 to 1.0]

        image_pass2 = torch.cat((image_patches_unfolded_pass1, (mask_pass1.to(device) * 2 - 1)), dim=1)
        output_pass2 = convolution_pass2(image_pass2)

        image_stride_h = image_patch_stride[0]
        image_stride_w = image_patch_stride[1]

        output_mask = output_pass2.permute(1, 0, 2, 3)
        reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                     image_patch_size[0], image_patch_size[1])

        merged_tensor = torch.zeros([1, h, w]).to(device)

        if gauss_count is None:
            gauss_count = torch.zeros([1, image_target_size[0], image_target_size[1]]).to(device)
            for ch in range(reshaped_refolded_patches.size(0)):
                for i in range(reshaped_refolded_patches.size(1)):
                    for j in range(reshaped_refolded_patches.size(2)):
                        x_start, y_start = i * image_stride_h, j * image_stride_w
                        gauss_count[ch,
                        x_start:x_start + image_patch_size[0],
                        y_start:y_start + image_patch_size[1]] += gauss_patch

        for ch in range(reshaped_refolded_patches.size(0)):
            for i in range(reshaped_refolded_patches.size(1)):
                for j in range(reshaped_refolded_patches.size(2)):
                    x_start, y_start = i * image_stride_h, j * image_stride_w

                    merged_tensor[
                    ch,
                    x_start:x_start + image_patch_size[0],
                    y_start:y_start + image_patch_size[1]] += \
                        (reshaped_refolded_patches[ch, i, j] * gauss_patch)

        merged_tensor /= gauss_count
        merged_tensor = torch.clamp_max(merged_tensor, 1)

        binary_mask_vessels = magic_wand_mask_selection_faster(merged_tensor).to(torch.float32)
        img_out = to_pil_transform(binary_mask_vessels)

        name, extension = os.path.splitext(sample.name)
        if extension.endswith("jpg"):
            extension = ".png"
        img_out.save(os.path.join(args.image_dir_mask_out, name + ".mask" + extension))


if __name__ == '__main__':
    main()
