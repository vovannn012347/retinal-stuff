from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import cv2
import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class DictConfig(object):
    """Creates a Config object from a dict
       such that object attributes correspond to dict keys.
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)


def save_nerve_definitor(fname, definitor, optimizer_pass, n_iter, config):
    state_dicts = {'nerve_definitor': definitor.state_dict(),
                   'adam_opt_nerve_definitor': optimizer_pass.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_2_convolution_states(fname, convolution_pass1, convolution_pass2, optimizer_pass1, optimizer_pass2, n_iter, config):
    state_dicts = {'convolution_pass1': convolution_pass1.state_dict(),
                   'convolution_pass2': convolution_pass2.state_dict(),
                   'adam_opt_pass1': optimizer_pass1.state_dict(),
                   'adam_opt_pass2': optimizer_pass2.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_2_convolution_states_2(fname, convolution_pass, optimizer_pass, n_iter, config):
    state_dicts = {'convolution_pass': convolution_pass.state_dict(),
                   'adam_opt_pass': optimizer_pass.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_nerve_classifier(fname, nerve_classifier_pass, optimizer, n_iter, config):
    state_dicts = {'nerve_classifier': nerve_classifier_pass.state_dict(),
                   'adam_opt_nerve_classifier': optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")

def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


def gaussian_2d(shape, center=None, sigma=1, min_value=0.0, max_value=1.0):
    if center is None:
        center = [shape[0] // 2 - 1, shape[1] // 2 - 1]  # Center of the array

    x = torch.arange(shape[0]).float()
    y = torch.arange(shape[1]).float()
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Calculate distance from the center
    dist = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Calculate the Gaussian distribution
    gaussian = torch.exp(-dist ** 2 / (2 * sigma ** 2))

    upper_left = gaussian[:(center[0] + 1), :(center[1] + 1)]
    upper_right = torch.flip(upper_left, dims=[1])
    lower_left = torch.flip(upper_left, dims=[0])
    lower_right = torch.flip(lower_left, dims=[1])

    gauss_max = upper_left.max()
    gauss_min = upper_left.min()

    gaussian[:(center[0] + 1), -(center[1] + 1):] = upper_right
    gaussian[-(center[0] + 1):, :(center[1] + 1)] = lower_left
    gaussian[-(center[0] + 1):, -(center[1] + 1):] = lower_right

    gauss_multiplier = (max_value - min_value) / (gauss_max - gauss_min)
    gaussian = min_value + (gaussian - gauss_min) * gauss_multiplier

    return gaussian


# requires image tensor with [-1, 1] values
def run_retina_cnn(image_tensor, retina_pass1, retina_pass2, image_patch_size, image_patch_stride,
                   device=None, apply_gauss=True):
    image_patches_unfolded_pass1 = image_tensor.unfold(1, image_patch_size[0], image_patch_size[0])
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.unfold(2, image_patch_size[1],
                                                                       image_patch_size[1])
    _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass1.size()
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.contiguous()
    image_patches_unfolded_pass1 = (image_patches_unfolded_pass1
                                    .view(image_tensor.size(0), -1,
                                          image_patch_size[0],
                                          image_patch_size[1]))
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.permute(1, 0, 2, 3)

    output_pass1 = retina_pass1(image_patches_unfolded_pass1)
    if apply_gauss is True:
        gauss = gaussian_2d([image_patch_size[0], image_patch_size[1]], min_value=0.65, max_value=1.0).unsqueeze(
            0)
        output_pass1 = output_pass1 * gauss

    recombined = torch.zeros_like(output_pass1)
    if device is not None:
        recombined = recombined.to(device)

    recombined.copy_(output_pass1)
    recombined = recombined.permute(1, 0, 2, 3).view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                     image_patch_size[0], image_patch_size[1])
    recombined = recombined.permute(0, 1, 3, 2, 4).reshape(1, mask_unfold_count_h * image_patch_size[0],
                                                           mask_unfold_count_w * image_patch_size[1])

    image_pass2 = torch.cat((image_tensor, (recombined * 2 - 1)), dim=0)

    image_patches_unfolded_pass2 = image_pass2.unfold(1, image_patch_size[0], image_patch_stride[0])
    image_patches_unfolded_pass2 = image_patches_unfolded_pass2.unfold(2, image_patch_size[1],
                                                                       image_patch_stride[1])
    _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass2.size()
    image_patches_unfolded_pass2 = image_patches_unfolded_pass2.contiguous()
    image_patches_unfolded_pass2 = image_patches_unfolded_pass2.view(image_tensor.size(0) + 1, -1,
                                                                     image_patch_size[0], image_patch_size[1])

    image_patches_unfolded_pass2 = image_patches_unfolded_pass2.permute(1, 0, 2, 3)

    output = retina_pass2(image_patches_unfolded_pass2)

    image_stride_h = image_patch_stride[0]
    image_stride_w = image_patch_stride[1]
    output_mask = output.permute(1, 0, 2, 3)
    reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                 image_patch_size[0], image_patch_size[1])

    merged_tensor = torch.zeros([1, image_tensor.size(1), image_tensor.size(2)])
    if device is not None:
        merged_tensor = merged_tensor.to(device)

    eps = 0.1

    for ch in range(reshaped_refolded_patches.size(0)):
        for i in range(reshaped_refolded_patches.size(1)):
            for j in range(reshaped_refolded_patches.size(2)):
                x_start, y_start = i * image_stride_h, j * image_stride_w

                # select all values
                temp_mask = ((merged_tensor[ch,
                              x_start:x_start + image_patch_size[0],
                              y_start:y_start + image_patch_size[1]]
                              < reshaped_refolded_patches[ch, i, j]) & (
                                     reshaped_refolded_patches[ch, i, j] > eps))

                temp_mask[:1, :] = False
                temp_mask[-1:, :] = False
                temp_mask[:, :1] = False
                temp_mask[:, -1:] = False

                merged_tensor[ch,
                x_start:x_start + image_patch_size[0],
                y_start:y_start + image_patch_size[1]][temp_mask] = (
                    reshaped_refolded_patches)[ch, i, j][temp_mask]

    return merged_tensor


def run_retina_cnn_2(image_tensor, retina_pass1, image_patch_size, image_patch_stride,
                     device=None, apply_gauss=True):
    image_patches_unfolded_pass1 = image_tensor.unfold(1, image_patch_size[0], image_patch_stride[0])
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.unfold(2, image_patch_size[1], image_patch_stride[1])
    _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass1.size()
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.contiguous()
    image_patches_unfolded_pass1 = (image_patches_unfolded_pass1
                                    .view(image_tensor.size(0), -1,
                                          image_patch_size[0],
                                          image_patch_size[1]))
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.permute(1, 0, 2, 3)

    output_pass1 = retina_pass1(image_patches_unfolded_pass1)

    channels, h, w = image_tensor.shape
    image_stride_h = image_patch_stride[0]
    image_stride_w = image_patch_stride[1]

    output_mask = output_pass1.permute(1, 0, 2, 3)
    reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                 image_patch_size[0], image_patch_size[1])

    merged_tensor = torch.zeros([1, h, w]).to(device)
    gauss_count = torch.zeros_like(merged_tensor)
    gauss_patch = gaussian_2d([image_patch_size[0], image_patch_size[1]],
                              min_value=0.5, max_value=1.6, sigma=20)

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

                # select all values
                # temp_mask = reshaped_refolded_patches[ch, i, j] > 0.05

                merged_tensor[
                    ch,
                    x_start:x_start + image_patch_size[0],
                    y_start:y_start + image_patch_size[1]] += (
                            reshaped_refolded_patches[ch, i, j] * gauss_patch)
                # (reshaped_refolded_patches[ch, i, j] * temp_mask * gauss_patch)

    merged_tensor /= gauss_count
    merged_tensor = torch.clamp_max(merged_tensor, 1)

    merged_tensor = magic_wand_mask_selection(merged_tensor).to(torch.float32)

    """target_color = 0
    tolerance = 0.23

    # binary_mask = abs(merged_tensor - target_color) < tolerance
    merged_tensor = (abs(merged_tensor - target_color) > tolerance).to(torch.float32)"""

    return merged_tensor


def magic_wand_mask_selection_batch(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): batch of 1 channel tensors that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """

    output_mask = torch.zeros_like(image_tensor, dtype=torch.bool)
    for i in range(output_mask.size(0)):
        output_mask[i] = magic_wand_mask_selection(image_tensor[i], upper_multiplier, lower_multipleir)

    return output_mask


def magic_wand_mask_selection(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    # part 1: get above zero pixel values
    flat_image = image_tensor.flatten()

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()
    histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

    # part 2: get starting tolerance and starting pixel value
    bin_width = (max_pixel - min_pixel) / bin_count
    non_zero_indices = torch.nonzero(histogram, as_tuple=False)

    first_tolerance = upper_multiplier

    upper_bound_bin_index1 = non_zero_indices[-1].item()
    lower_bound_bin_index1 = int(upper_bound_bin_index1 * (1 - first_tolerance))

    # upper_bound = (upper_bound_bin_index + 1) * bin_width
    lower_bound = lower_bound_bin_index1 * bin_width

    # part 3: make starting global selection
    first_selection = image_tensor > lower_bound

    # result_image = first_selection.to(torch.float32)
    # result_image = TF.to_pil_image(result_image.squeeze().cpu(), mode="L")
    # result_image.save(f"{debug_dir}/test1.png")

    # part 4: replace selected pixel values with the lowest value from selected pixels
    image_tensor_editable = torch.clone(image_tensor)
    image_tensor_editable[first_selection] = lower_bound

    # result_image = TF.to_pil_image(image_tensor_editable.squeeze().cpu(), mode="L")
    # result_image.save(f"{debug_dir}/test2.png")

    # part 5: get second tolerance value
    # from the lowest color value in part 2 to total lowest value that is above 0 color
    lower_bound_bin_index2 = int(lower_bound_bin_index1 * lower_multipleir)
    if lower_bound_bin_index2 < 3:
        lower_bound_bin_index2 = 3

    lower_bound2 = lower_bound_bin_index2 * bin_width

    # part 6: make second selection starting from first selection locations using second tolerance
    output_mask = torch.zeros_like(first_selection, dtype=torch.bool)
    for i in range(first_selection.size(1)):
        for j in range(first_selection.size(2)):

            if first_selection[0, i, j].item() is True and output_mask[0, i, j].item() is False:
                to_test = [(i, j)]

                while to_test:
                    x, y = to_test.pop()

                    # Check if the pixel is already in the mask
                    if output_mask[0, x, y].item():
                        continue

                    # Get the value of the current pixel
                    pixel_value = image_tensor_editable[0, x, y]

                    # If the pixel value is within the tolerance range, include it in the mask
                    if lower_bound2 <= pixel_value:  # <= upper_bound2:
                        # actually we are not intrested in upper bound as we go top-down
                        output_mask[0, x, y] = True

                        # Explore neighboring pixels (4-connectivity: top, bottom, left, right)
                        if (x > 0
                                and output_mask[0, x - 1, y].item() is False):
                            to_test.append((x - 1, y))  # Top neighbor
                        if (x < image_tensor.shape[1] - 1
                                and output_mask[0, x + 1, y].item() is False):
                            to_test.append((x + 1, y))  # Bottom neighbor
                        if (y > 0
                                and output_mask[0, x, y - 1].item() is False):
                            to_test.append((x, y - 1))  # Left neighbor
                        if (y < image_tensor.shape[2] - 1
                                and output_mask[0, x, y + 1].item() is False):
                            to_test.append((x, y + 1))  # Right neighbor

    # for i, count in enumerate(histogram):
    #     lower_bound = min_pixel + i * bin_width
    #     upper_bound = lower_bound + bin_width
    #     print(f"Bin {i + 1}: Range [{lower_bound:.2f}, {upper_bound:.2f}], Count: {int(count)}")

    return output_mask

def magic_wand_mask_selection_batch_faster(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    if image_tensor.dim() != 4 or image_tensor.size(1) != 1:  # RGB
        raise Exception("invalid image_tensor dimensions")

    bounds = []
    masks = []
    image_tensor_wand = torch.clone(image_tensor)

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()

    for image_i in range(image_tensor.size(0)):
        # part 1: get above zero pixel values
        flat_image = image_tensor[image_i].flatten()

        histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

        # part 2: get starting tolerance and starting pixel value
        bin_width = (max_pixel - min_pixel) / bin_count
        non_zero_indices = torch.nonzero(histogram, as_tuple=False)

        first_tolerance = upper_multiplier

        first_bound_bin_index = int(non_zero_indices[-1].item() * (1 - first_tolerance))
        first_bound = first_bound_bin_index * bin_width

        # part 3: make starting global selection

        mask = image_tensor[image_i] > first_bound
        masks.append(mask)

        # part 4: replace selected pixel values with the lowest value from selected pixels
        image_tensor_wand[image_i][mask] = first_bound

        # part 5: get second tolerance value
        # from the lowest color value in part 2 to total lowest value that is above 0 color
        second_bound_bin_index = int(first_bound_bin_index * lower_multipleir)
        if second_bound_bin_index < 3:
            second_bound_bin_index = 3
            if second_bound_bin_index >= first_bound_bin_index:
                second_bound_bin_index = first_bound_bin_index - 1

        if second_bound_bin_index < 0:
            second_bound_bin_index = 0

        second_bound = second_bound_bin_index * bin_width
        bounds.append(second_bound)

    bounds = torch.tensor(bounds).view(-1, 1, 1, 1)

    masks = torch.stack(masks)
    diff_map = image_tensor_wand >= bounds

    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32,
                          device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    max_iters = int((diff_map.size(2) * diff_map.size(3)) / 4)

    for _ in range(max_iters):
        dilated_mask = nnF.conv2d(masks.float(), kernel, padding=1).bool()

        # Mask update: keep pixels within threshold and add to current mask
        new_masks = dilated_mask & diff_map

        # Stop if no new pixels are added
        if torch.equal(new_masks, masks):
            break

        # Update mask with new selection
        masks = new_masks

    return masks


def magic_wand_mask_selection_faster(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    if image_tensor.dim() != 3 or image_tensor.size(0) != 1:  # RGB
        raise Exception("invalid image_tensor dimensions")

    # part 1: get above zero pixel values
    flat_image = image_tensor.flatten()

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()
    histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

    # part 2: get starting tolerance and starting pixel value
    bin_width = (max_pixel - min_pixel) / bin_count
    non_zero_indices = torch.nonzero(histogram, as_tuple=False)

    first_tolerance = upper_multiplier

    first_bound_bin_index = int(non_zero_indices[-1].item() * (1 - first_tolerance))
    first_bound = first_bound_bin_index * bin_width

    # part 3: make starting global selection
    mask = image_tensor > first_bound

    # part 4: replace selected pixel values with the lowest value from selected pixels
    image_tensor_wand = torch.clone(image_tensor)
    image_tensor_wand[mask] = first_bound

    # part 5: get second tolerance value
    # from the lowest color value in part 2 to total lowest value that is above 0 color
    lower_bound_bin_index = int(first_bound_bin_index * lower_multipleir)
    if lower_bound_bin_index < 3:
        lower_bound_bin_index = 3
        if lower_bound_bin_index >= first_bound_bin_index:
            lower_bound_bin_index = first_bound_bin_index - 1

    if lower_bound_bin_index < 0:
        lower_bound_bin_index = 0

    lower_bound = lower_bound_bin_index * bin_width

    diff_map = ((image_tensor_wand - lower_bound) >= 0).squeeze()

    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32,
                          device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    max_iters = int((mask.size(1) * mask.size(2)) / 4)

    #mask = mask.squeeze(0)

    for _ in range(max_iters):
        dilated_mask = (nnF.conv2d(mask.float().unsqueeze(0), kernel, padding=1)
                        .squeeze().bool())

        # Mask update: keep pixels within threshold and add to current mask
        new_mask = dilated_mask & diff_map

        # Stop if no new pixels are added
        if torch.equal(new_mask, mask):
            break

        # Update mask with new selection
        mask = new_mask

    return mask.unsqueeze(0)

class RandomGreyscale(torch.nn.Module):
    """Monochromes given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
            T.ToTensor()  # Convert PIL image to tensor
        ])

    def forward(self, img):

        """
        Args:
            img (PIL Image or Tensor): Image to be made monochrome.

        Returns:
            PIL Image or Tensor: image made monochrome or not.
        """
        if self.p >= 1 or torch.rand(1) < self.p:
            if isinstance(img, Image.Image):
                return img.convert("L")
            else:
                return self.transform(img)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def histogram_equalization_lab(image):

    image_np = np.array(image)

    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2Lab)

    # Split into L, a, b channels
    l, a, b = cv2.split(lab)

    # Apply histogram equalization on the L channel
    l_eq = cv2.equalizeHist(l)

    # Merge the equalized L channel with a and b channels
    lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR (RGB) color space
    image_equalized = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    return Image.fromarray(image_equalized)


def histogram_equalization_hsv_s(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s_eq = cv2.equalizeHist(s)
    hsv_eq = cv2.merge((h, s_eq, v))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)


def histogram_equalization_hsv_v(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)


def apply_clahe_rgb(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Split the image into R, G, B channels
    channels = cv2.split(image_np)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = [clahe.apply(channel) for channel in channels]

    # Merge the channels back together
    image_clahe = cv2.merge(channels)

    # Convert back to PIL Image
    return Image.fromarray(image_clahe)


def apply_clahe_lab(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Convert RGB to LAB color space
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    # Split into L, A, and B channels
    L, A, B = cv2.split(lab_image)

    # Apply CLAHE to the L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L = clahe.apply(L)

    # Merge the channels back and convert to RGB
    lab_image = cv2.merge((L, A, B))
    image_clahe_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Convert back to PIL Image
    return Image.fromarray(image_clahe_rgb)

