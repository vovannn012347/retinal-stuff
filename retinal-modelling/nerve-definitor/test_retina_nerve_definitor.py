import argparse
import os.path
import time

from PIL import Image
import torch
import torchvision.transforms as tvtransf
import torch.nn.functional as tochfunc

from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from model.retina_classifier_networks import FcnskipNerveDefinitor2
from utils.retinaldata import get_bounding_box_fast, open_image, get_bounding_box_rectanglified, \
    extract_objects_with_contours_np_cv2

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-stuff-definitor/checkpoints/states_02_01_2025.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image", type=str,
                    #default="training-data/preprocess-output/kaggle-images/1222_right.jpg",
                    default="training-data/preprocess-output/kaggle-images/1264_right.jpg",
                    help="path to the image file")
parser.add_argument("--out-dir", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output",
                    help="path to the output mask file")

img_shapes = [576, 576, 3]
load_mode = "RGB"

def split_by_position(string, position):
    return [string[:position], string[position:]]


'''def measure_elipcicity_threshold(image, bbox):
    # we assume image is greyscale single channel
    # Unpack bounding box dimensions
    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox

    # let's assume that whole image is good enough?
    # should return something else
    if bbox_top == 0 & bbox_left == 0 & bbox_bottom == image.size(1) & bbox_right == image.size(2):
        return 1

    # Apply the mask to the image (ensure image size matches bbox)
    masked_image = image[:, bbox_top:bbox_bottom, bbox_left:bbox_right]

    #measure elipsicity
    coords = torch.nonzero(masked_image, as_tuple=False).float()
    n_points = coords.size(0)

    if n_points == 0:
        return 0

    center = coords.sum(dim=0) / n_points

    coords -= center

    covariance_matrix = torch.matmul(coords.T, coords) / (n_points - 1)

    # Use SVD to get eigenvalues for the principal axes (more efficient than `eigh`)
    _, singular_values, _ = torch.svd(covariance_matrix)

    singular_values, _ = torch.sort(singular_values, descending=True)

    # Calculate Eccentricity and Aspect Ratio
    major_axis_length = torch.sqrt(singular_values[0])
    minor_axis_length = torch.sqrt(singular_values[1])
    # Ellipticity measures
    eccentricity = torch.sqrt(1 - (minor_axis_length**2 / major_axis_length**2))
    aspect_ratio = minor_axis_length / major_axis_length

    bbox_width = bbox_right - bbox_left # img_bbox[2] - img_bbox[0]
    bbox_height = bbox_bottom - bbox_top # img_bbox[3] - img_bbox[1]

    # Calculate the center and radius lengths for the ellipse
    center_y, center_x = bbox_height / 2, bbox_width / 2
    radius_y, radius_x = bbox_height / 4, bbox_width / 4

    # Create a grid of coordinates
    y = torch.arange(bbox_height).view(-1, 1).expand(bbox_height, bbox_width)
    x = torch.arange(bbox_width).view(1, -1).expand(bbox_height, bbox_width)

    # Create the elliptical mask using the ellipse equation
    ellipse_mask = (((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2) <= 1

    # Count pixels within the ellipse and above the threshold
    ellipse_pixels = masked_image[ellipse_mask.unsqueeze(0)]
    pixels_above_threshold = (ellipse_pixels > 0).to(torch.float32).sum().item()

    # Calculate the percentage
    total_ellipse_pixels = ellipse_mask.sum().item()
    fraction_above_threshold = (pixels_above_threshold / total_ellipse_pixels)

    mult1 = 1 / (1 + eccentricity)
    mult2 = 1 / (1 + aspect_ratio)
    mult3 = mult2 / mult1

    return fraction_above_threshold * mult3'''


'''def get_bounding_box_fast(image_tensor):
    # Check for non-zero values along each axis to find the bbox
    img = image_tensor.squeeze(0)
    rows = torch.any(img > 0, dim=1)  # True for rows with non-zero pixels
    cols = torch.any(img > 0, dim=0)  # True for columns with non-zero pixels

    # Get min and max of non-zero rows and columns
    min_y, max_y = torch.where(rows)[0][[0, -1]]
    min_x, max_x = torch.where(cols)[0][[0, -1]]

    # left, top, right, bottom
    bbox = (min_x.item(), min_y.item(), max_x.item(), max_y.item())
    return bbox'''



def main():

    args = parser.parse_args()
    time0 = time.time()
    # set up network
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    definitor = FcnskipNerveDefinitor2(num_classes=1).to(device)
    definitor.load_state_dict(convolution_state_dict['nerve_definitor'])

    print(f"input file at: {args.image}")

    '''pil_image = open_image(sample.path)  # Image.open(sample.path).convert('RGB')  # 3 channel

    # pil_image_processed = histogram_equalization_hsv_s(pil_image)
    # pil_image_processed = apply_clahe_lab(pil_image_processed)
    # pil_image_processed = pil_image_processed.convert("L")
    pil_image_processed = pil_image.convert("L")

    tensor_image = T.ToTensor()(pil_image_processed)

    tensor_image = T.Resize(max(img_shapes[:2]), antialias=True)(tensor_image)
    pil_image = pil_image.resize(img_shapes[:2])
    channels, h, w = tensor_image.shape
    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]'''

    image_name = os.path.basename(args.image) # file name
    image_name, image_ext = os.path.splitext(image_name) # name and extension

    pil_image_origin = open_image(args.image)  # Image.open(args.image).convert('RGB')  # 3 channel

    pil_image_processed = histogram_equalization_hsv_s(pil_image_origin)
    pil_image_processed = apply_clahe_lab(pil_image_processed)
    pil_image_processed = pil_image_processed.convert("L")
    # pil_image_processed = pil_image_origin.convert("L")

    tensor_image = tvtransf.ToTensor()(pil_image_processed)
    tensor_image = tvtransf.Resize(img_shapes[:2], antialias=True)(tensor_image)
    pil_image_origin = pil_image_origin.resize(img_shapes[:2])
    channels, h, w = tensor_image.shape
    # training is done on smaller resolution image
    tensor_image = tochfunc.interpolate(tensor_image.unsqueeze(0),
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)

    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    tensor_image = tensor_image.unsqueeze(0)
    output = definitor(tensor_image).squeeze(0)

    to_pil_transform = tvtransf.ToPILImage(mode='L')

    output[output < 0.09] = 0
    # output = torch.clamp(output, 0.09, 1)
    output_wand_selected = (magic_wand_mask_selection_faster(output, upper_multiplier=0.15, lower_multipleir=0.3)
                              .to(torch.float32))

    channels_bb, h_bb, w_bb = output_wand_selected.shape
    split_tensors = extract_objects_with_contours_np_cv2(output_wand_selected)

    out_filenames = []

    for split_idx, tensor in enumerate(split_tensors):

        '''output_selected = to_pil_transform(tensor)
        mask_path_out = os.path.join(args.out_dir, f"{image_name}_mask_{split_idx}{image_ext}")
        output_selected.save(mask_path_out)'''

        img_bbox = get_bounding_box_fast(tensor)  # left, top, right, bottom

        expand_constant = 0.2
        bb_w = img_bbox[2] - img_bbox[0]
        bb_h = img_bbox[3] - img_bbox[1]
        img_bbox2 = (img_bbox[0] - bb_w * expand_constant) / w_bb, (img_bbox[1] - bb_h * expand_constant) / h_bb, (
                    img_bbox[2] + bb_w * expand_constant) / w_bb, (img_bbox[3] + bb_h * expand_constant) / h_bb
        img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
        img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

        img_bbox4 = get_bounding_box_rectanglified(img_bbox4, h, w)

        if (img_bbox4[2] - img_bbox4[0]) > 1 and (img_bbox4[3] - img_bbox4[1]) > 1:
            pil_image_cropped = pil_image_origin.crop(img_bbox4)
            # pil_image_cropped.save(args.out_cropped)
            out_filename = f"{image_name}_cropped_{split_idx}{image_ext}"
            image_cropped_path_out = os.path.join(args.out_dir, out_filename)
            pil_image_cropped.save(image_cropped_path_out)
            out_filenames.append(out_filename)

    dt = time.time() - time0

    print(f"@timespan: {dt} s")


if __name__ == '__main__':
    main()

