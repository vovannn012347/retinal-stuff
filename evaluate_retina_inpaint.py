import argparse
import os
from PIL import Image
import torch
import math
import glob
import torchvision.transforms as T
import utils.misc_inpaint as misc
import torch.nn as nn
from utils.retinaldata import is_image_file
import torchvision.transforms.functional as t_tf_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Evaluate inpainting - creates plot that describes differences between'
                                             ' images via numerical evaluations')

parser.add_argument("--images", type=str,
                    default="training-data/retina-detection/training-images-batch-processing-output",
                    help="path to the source image directory")
parser.add_argument("--masks", type=str,
                    default="training-data/retina-detection/training-masks-batch-processing-output",
                    help="path to the mask directory")
parser.add_argument("--infill-images", type=str,
                    default="training-data/retina-infill/batch-output",
                    help="path to the infilled images directory")
parser.add_argument("--out-gauss", type=str,
                    default="training-data/retina-infill/gauss-blurred",
                    help="path for the gaus blurred images directory")


class UniformMaskedRadialSpreader(nn.Module):
    def __init__(self, radius, wand_tolerance, gauss_sigma=30, device='cpu'):
        super(UniformMaskedRadialSpreader, self).__init__()

        self.training = False

        size = 2 * radius + 1
        center = radius
        tensor = torch.zeros((size, size), dtype=torch.float32)

        center2 = center ** 2
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= center2:
                    tensor[i, j] = 1.0

        channels = 3
        self.kernel_1ch = tensor.unsqueeze(0).unsqueeze(0)
        self.kernel_3ch = self.kernel_1ch.expand(channels, 1, size, size)

        self.conv_1ch = torch.nn.Conv2d(1, 1, kernel_size=size, groups=1, bias=False, padding=size // 2)
        self.conv_1ch.weight.data = self.kernel_1ch
        self.conv_1ch.weight.requires_grad = False

        self.conv_3ch = torch.nn.Conv2d(channels, channels,
                                        kernel_size=size, groups=channels, bias=False, padding=size // 2)
        self.conv_3ch.weight.data = self.kernel_3ch
        self.conv_3ch.weight.requires_grad = False

        self.gaussian_conv = misc.get_gaussian_conv2d(device, channels, size, gauss_sigma)

        self.eps = 1e-6
        self.tolerance = wand_tolerance

    # batch images are expected to be [-1.0 ... 1.0]
    # masks are expected to be 0.0/1.0
    def forward(self, batch_images: torch.Tensor, batch_mask: torch.Tensor):

        batch_dark_area_mask = torch.zeros_like(batch_mask)

        start_x = 0
        start_y = 0
        for mask_index in range(batch_images.shape[0]):
            mask = batch_dark_area_mask[mask_index, :, :, :]
            image = batch_images[mask_index, :, :, :]
            stack = [(start_x, start_y),
                     (image.shape[1] - 1 - start_x, start_y),
                     (image.shape[1] - 1 - start_x, image.shape[2] - 1 - start_y),
                     (start_x, image.shape[2] - 1 - start_y)]
            start_color = image[:, start_x, start_y].clone()
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[2]:
                    continue
                if mask[0, x, y] > 0:
                    continue

                current_color = image[:, x, y]
                if torch.sqrt(torch.sum((current_color - start_color) ** 2)) > self.tolerance:
                    continue

                mask[0, x, y] = 1.0  # Mark the position as visited

                # Add neighboring positions to the stack
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

        batch_mask_inverted = 1 - batch_mask
        batch_images_normalized = (batch_images + 1) / 2
        batch_blur_mask = (batch_mask_inverted - batch_dark_area_mask).clamp(min=0.0)
        batch_image_mask = (1 - batch_dark_area_mask)

        batch_images_pre_blur = batch_images_normalized * batch_blur_mask

        batch_mask_blurred = self.conv_1ch(batch_mask_inverted)
        batch_image_blurred = self.conv_3ch(batch_images_pre_blur)

        batch_mask_missed = (batch_mask_blurred < self.eps).float()  # important to care about 0 division

        batch_mask_blurred_divisor = batch_mask_blurred * batch_mask
        batch_image_blurred_divisible = batch_image_blurred * (batch_mask - batch_mask_missed)

        # contains only blurred result excluding unmasked areas not toucnhed by blur
        batch_image_blurred_result = ((batch_image_blurred_divisible /
                                      (batch_mask_blurred_divisor + batch_mask_inverted + batch_mask_missed))
                                      * batch_image_mask)

        batch_images_nonblurred_result = batch_images_normalized * (batch_mask_inverted + batch_mask_missed)

        batch_images_result = batch_image_blurred_result + batch_images_nonblurred_result

        batch_images_result = (batch_images_result * 2 - 1).clamp(min=-1.0, max=1.0)

        return batch_images_result

def get_image_with_name_any_extension(folder_path, image_name):
    search_pattern = os.path.join(folder_path, f"{image_name}.*")

    image_files = glob.glob(search_pattern)

    # Check if any image file is found
    if image_files:
        return image_files[0]  # Return the first found image (if multiple, modify as needed)
    else:
        return None


def SSIM_difference(image1, image2):
    return ssim(image1, image2, data_range=1.0).item()


mse_loss = nn.MSELoss()
def PSNR_difference(image1, image2):

    return 10 * math.log10(mse_loss(image1, image2))


def main():

    args = parser.parse_args()

    gaussian_blur_convo = UniformMaskedRadialSpreader(19, 0.25)

    folder_path = args.images
    mask_folder_path = args.masks
    modified_image_output_folder = args.infill_images
    output_gauss = args.out_gauss

    ssim_values = []

    psnr_values = []

    for sample in [entry for entry in os.scandir(folder_path) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        # load image and mask
        name, extension = os.path.splitext(sample.name)
        if extension.endswith("jpg"):
            extension = ".png"

        mask_name = get_image_with_name_any_extension(mask_folder_path, name)
        if mask_name is None:
            print(f"Mask not found, skipping")
            continue

        infilled_name = get_image_with_name_any_extension(modified_image_output_folder, name)
        if mask_name is None:
            print(f"Mask not found, skipping")
            continue

        # load image and mask
        image = Image.open(sample.path).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        image_infilled = Image.open(infilled_name).convert('RGB')

        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        image_infilled = T.ToTensor()(image_infilled)

        image_gauss_spread = image * 2 - 1
        image_gauss_spread = (gaussian_blur_convo(image_gauss_spread.unsqueeze(0), mask.unsqueeze(0)) + 1) / 2
        image_gauss_spread = image_gauss_spread.squeeze()

        img_out = t_tf_fn.to_pil_image(image_gauss_spread, mode="RGB")
        img_out.save(os.path.join(output_gauss, name + extension))

        image_to_spread = SSIM_difference(image.unsqueeze(0), image_gauss_spread.unsqueeze(0))
        image_to_infill = SSIM_difference(image.unsqueeze(0), image_infilled.unsqueeze(0))
        spread_to_infill = SSIM_difference(image_gauss_spread.unsqueeze(0), image_infilled.unsqueeze(0))

        print(f"{name} SSIM: image/infill, image/spread, spread/infill: {image_to_infill}, {image_to_spread}, {spread_to_infill}")

        ssim_values.append([image_to_spread, image_to_infill, spread_to_infill])

        image_to_spread = PSNR_difference(image, image_gauss_spread)
        image_to_infill = PSNR_difference(image, image_infilled)
        spread_to_infill = PSNR_difference(image_gauss_spread, image_infilled)

        print(f"{name} PSNR: image/infill, image/spread, spread/infill: {image_to_infill}, {image_to_spread}, {spread_to_infill}")

        psnr_values.append([image_to_spread, image_to_infill, spread_to_infill])

    ssim_values_sorted = sorted(ssim_values, key=lambda x: x[0], reverse=True)

    # Separate the sorted values for plotting
    ssim1_sorted = [x[0] for x in ssim_values_sorted]
    ssim2_sorted = [x[1] for x in ssim_values_sorted]
    ssim3_sorted = [x[2] for x in ssim_values_sorted]

    # Create an index array for the x-axis (sorted by ssim1)
    x = list(range(1, len(ssim_values) + 1))

    # Plotting the sorted SSIM values
    plt.figure(figsize=(10, 6))
    plt.plot(x, ssim1_sorted, label='image/spread', marker='o')
    plt.plot(x, ssim2_sorted, label='image/infill', marker='o')
    plt.plot(x, ssim3_sorted, label='spread/infill', marker='o')

    # Labels and Title
    plt.title('Ssim stuff', fontsize=14)
    plt.xlabel('Image by similarity', fontsize=12)
    plt.ylabel('SSIM Value', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.savefig('ssim_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution (300 DPI)

    # Optionally display the plot (if you want to see it)
    plt.show()

    psnr_values_sorted = sorted(psnr_values, key=lambda x: x[0], reverse=True)

    # Separate the sorted values for plotting
    psnr1_sorted = [x[0] for x in psnr_values_sorted]
    psnr2_sorted = [x[1] for x in psnr_values_sorted]
    psnr3_sorted = [x[2] for x in psnr_values_sorted]

    # Create an index array for the x-axis (sorted by psnr1)
    x = list(range(1, len(psnr_values) + 1))

    # Plotting the sorted psnr values
    plt.figure(figsize=(10, 6))
    plt.plot(x, psnr1_sorted, label='image/spread', marker='o')
    plt.plot(x, psnr2_sorted, label='image/infill', marker='o')
    plt.plot(x, psnr3_sorted, label='spread/infill', marker='o')

    # Labels and Title
    plt.title('psnr stuff', fontsize=14)
    plt.xlabel('Image by similarity', fontsize=12)
    plt.ylabel('psnr Value', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.savefig('psnr_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution (300 DPI)

    # Optionally display the plot (if you want to see it)
    plt.show()

if __name__ == '__main__':
    main()
