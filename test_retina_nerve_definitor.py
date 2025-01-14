import argparse
from PIL import Image
import torch
import torchvision.transforms as T
from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from model.retina_classifier_networks import FcnskipNerveDefinitor2
from utils.retinaldata import get_image_bbox, is_image_file, get_bounding_box_fast, pil_loader


parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-stuff-definitor/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image", type=str,
                    default="training-data/preprocess-output/histogram-hsv-s-clache-lab-no-inpaint/1222_right.jpg",
                    help="path to the image file")
parser.add_argument("--out-mask", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output/1222_right.mask.jpg",
                    help="path to the output mask file")
parser.add_argument("--out-cropped", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output/1222_right.jpg",
                    help="path for the output cropped file")

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


def open_image(image_path):
    pil_image = pil_loader(image_path, load_mode)
    img_bbox = get_image_bbox(pil_image)

    pil_image = pil_image.crop(img_bbox)
    return pil_image


def main():

    args = parser.parse_args()

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

    pil_image_origin = open_image(args.image) # Image.open(args.image).convert('RGB')  # 3 channel

    #pil_image_processed = histogram_equalization_hsv_s(pil_image_origin)
    #pil_image_processed = apply_clahe_lab(pil_image_processed)
    #pil_image_processed = pil_image_processed.convert("L")
    pil_image_processed = pil_image_origin.convert("L")

    tensor_image = T.ToTensor()(pil_image_processed)
    tensor_image = T.Resize(img_shapes[:2], antialias=True)(tensor_image)
    pil_image_origin = pil_image_origin.resize(img_shapes[:2])

    channels, h, w = tensor_image.shape
    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    tensor_image = tensor_image.unsqueeze(0)
    output = definitor(tensor_image).squeeze(0)

    to_pil_transform = T.ToPILImage(mode='L')

    output_selected = magic_wand_mask_selection_faster(output, upper_multiplier=0.15, lower_multipleir=0.4).to(torch.float32)
    channels_bb, h_bb, w_bb = output_selected.shape
    img_bbox = get_bounding_box_fast(output_selected)  # left, top, right, bottom

    output_selected = to_pil_transform(output_selected)
    output_selected.save(args.out_mask)

    #pil_image_origin.save(args.out_cropped)

    expand_constant = 2.0
    bb_w = img_bbox[2] - img_bbox[0]
    bb_h = img_bbox[3] - img_bbox[1]
    img_bbox2 = (img_bbox[0] - bb_w / expand_constant) / w_bb, (img_bbox[1] - bb_h / expand_constant) / h_bb, (
                img_bbox[2] + bb_w / expand_constant) / w_bb, (img_bbox[3] + bb_h / expand_constant) / h_bb
    img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
    img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

    pil_image_cropped = pil_image_origin.crop(img_bbox4)
    pil_image_cropped.save(args.out_cropped)

if __name__ == '__main__':
    main()

    # img_bbox = (0, 0, 0, 0)
    # fraction = 0.0
    # output_selected = output
    # channels_bb, h_bb, w_bb = output_selected.shape
    # output_temp = torch.zeros_like(output).to(device).copy_(output)

    # while fraction < 0.5:
    #    channels_bb, h_bb, w_bb = output_selected.shape
    #    img_bbox = get_bounding_box_fast(output_selected) # left, top, right, bottom
    #    fraction = measure_elipcicity_threshold(output_selected, img_bbox)
    #    output_temp[output_selected == 1] = 0


    #pil_image_cropped = pil_image_origin.crop(img_bbox4)
    #pil_image_cropped.save(args.out_cropped)

