import argparse
from PIL import Image
import torch
import os
import cv2
import numpy as np
import torchvision.transforms as T
from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from model.retina_classifier_networks import FcnskipNerveDefinitor2
from utils.retinaldata import get_image_bbox, is_image_file, get_bounding_box_fast, pil_loader

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-stuff-definitor/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image-dir", type=str,
                    #default="training-data/retina-infill/batch-output",
                    #default="training-data/retina-detection/training-images",
                    default="training-data/preprocess-output/histogram-hsv-s-clache-lab-no-inpaint",
                    help="path to the image file")
parser.add_argument("--image-dir-out", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output",
                    help="path for the output cropped file")
parser.add_argument("--image-mask-dir-out", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output_mask",
                    help="path to the output mask file")

img_shapes = [576, 576, 3]
load_mode = "RGB"

def open_image(image_path):
    pil_image = pil_loader(image_path, load_mode)
    img_bbox = get_image_bbox(pil_image)

    pil_image = pil_image.crop(img_bbox)
    return pil_image

def main():

    args = parser.parse_args()

    if not os.path.exists(args.image_dir_out):
        os.makedirs(args.image_dir_out)
    if not os.path.exists(args.image_mask_dir_out):
        os.makedirs(args.image_mask_dir_out)

    # set up network
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    definitor = FcnskipNerveDefinitor2(num_classes=1).to(device)
    definitor.load_state_dict(convolution_state_dict['nerve_definitor'])

    for sample in [entry for entry in os.scandir(args.image_dir) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        name, extension = os.path.splitext(sample.name)
        '''if extension.endswith("jpg"):
            extension = ".png"'''
        extension = ".jpg"

        pil_image = open_image(sample.path) #Image.open(sample.path).convert('RGB')  # 3 channel

        #pil_image_processed = histogram_equalization_hsv_s(pil_image)
        #pil_image_processed = apply_clahe_lab(pil_image_processed)
        #pil_image_processed = pil_image_processed.convert("L")
        pil_image_processed = pil_image.convert("L")

        tensor_image = T.ToTensor()(pil_image_processed)

        tensor_image = T.Resize(img_shapes[:2], antialias=True)(tensor_image)
        pil_image = pil_image.resize(img_shapes[:2])

        channels, h, w = tensor_image.shape
        tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
        #tensor_image = F.interpolate(tensor_image, scale_factor=0.5, mode='bilinear', align_corners=False)

        if tensor_image.size(0) == 1:
            tensor_image = torch.cat([tensor_image] * 3, dim=0)

        tensor_image = tensor_image.unsqueeze(0)
        output = definitor(tensor_image).squeeze(0)
        wand_output = magic_wand_mask_selection_faster(output, upper_multiplier=0.1, lower_multipleir=0.5).to(torch.float32)
        to_pil_transform = T.ToPILImage(mode='L')

        img_bbox = get_bounding_box_fast(wand_output)

        channels_bb, h_bb, w_bb = output.shape
        output = to_pil_transform(output)

        img_mask_name_path = os.path.join(args.image_mask_dir_out, name + ".mask" + extension)
        output.save(img_mask_name_path)

        #output = to_pil_transform(wand_output)
        #img_bbox = get_bounding_box_fast(output) # left, top, right, bottom

        bb_w = img_bbox[2] - img_bbox[0]
        bb_h = img_bbox[3] - img_bbox[1]

        expand_constant = 2.0
        img_bbox2 = (img_bbox[0] - bb_w / expand_constant) / w_bb, (img_bbox[1] - bb_h/expand_constant)/h_bb, (img_bbox[2] + bb_w / expand_constant) / w_bb, (img_bbox[3] + bb_h/expand_constant)/h_bb
        img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
        img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

        pil_image_cropped = pil_image.crop(img_bbox4)

        img_name_path = os.path.join(args.image_dir_out, name + extension)
        pil_image_cropped.save(img_name_path)

if __name__ == '__main__':
    main()
