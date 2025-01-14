import argparse
from PIL import Image
import torch
import os
import cv2
import numpy as np
import torchvision.transforms as T
from utils.misc_retina import gaussian_2d, magic_wand_mask_selection_faster
from utils.misc_retina import (apply_clahe_rgb, apply_clahe_lab,
                               histogram_equalization_hsv_v, histogram_equalization_hsv_s)

from utils.retinaldata import get_image_bbox, is_image_file

parser = argparse.ArgumentParser(description='preprocess images')

parser.add_argument("--image-input", type=str,
                    #default="training-data/retina-detection/training-images-batch-processing-output",
                    #default="training-data/retina-infill/batch-output",
                    #default="training-data/preprocess-output/kaggle-selection",
                    default="training-data/preprocess-output/local-images",
                    help="path to the image file")
parser.add_argument("--image-output", type=str,
                    default="training-data/preprocess-output/histogram-hsv-s-clache-lab-no-inpaint",
                    help="path to the output mask file")


def main():

    args = parser.parse_args()

    if not os.path.exists(args.image_output):
        os.makedirs(args.image_output)

    for sample in [entry for entry in os.scandir(args.image_input) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        name, extension = os.path.splitext(sample.name)
        #if extension.endswith("jpg"):
        extension = ".jpg"

        pil_image = Image.open(sample.path).convert('RGB')  # 3 channel
        pil_image = histogram_equalization_hsv_s(pil_image)
        pil_image = apply_clahe_lab(pil_image)

        img_mask_name_path = os.path.join(args.image_output, name + extension)
        pil_image.save(img_mask_name_path)
        #return

if __name__ == '__main__':
    main()
