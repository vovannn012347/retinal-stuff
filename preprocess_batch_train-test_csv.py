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
import pandas as pd

from utils.retinaldata import get_image_bbox, is_image_file

parser = argparse.ArgumentParser(description='preprocess images')

parser.add_argument("--file-input", type=str,
                    #default="training-data/preprocess-output/full_df.csv",
                    default="training-data/preprocess-output/local_hand_df.csv",
                    help="path to the csv dataset file")

parser.add_argument("--image-input", type=str,
                    default="training-data/preprocess-output/local-images",
                    help="path to the image files")

parser.add_argument("--output-image", type=str,
                    default="training-data/preprocess-output/local-glaucoma-atrophy",
                    help="path to the output dir")

parser.add_argument("--output-label", type=str,
                    default="training-data/retina-stuff-classifier/batch_test_retina_nerve_definitor_labels",
                    help="path to the output mask file")

'''parser.add_argument("--output-image-train", type=str,
                    default="training-data/retina-stuff-classifier/batch_test_retina_nerve_definitor_labels",
                    help="path to the output mask file")
parser.add_argument("--output-mask-train", type=str,
                    default="training-data/retina-stuff-classifier/batch_test_retina_nerve_definitor_labels",
                    help="path to the output mask file")

parser.add_argument("--output-image-test", type=str,
                    default="training-data/retina-stuff-classifier/batch_test_retina_nerve_definitor_labels",
                    help="path to the output mask file")
parser.add_argument("--output-mask-test", type=str,
                    default="training-data/retina-stuff-classifier/batch_test_retina_nerve_definitor_labels",
                    help="path to the output mask file")'''

def main():

    args = parser.parse_args()

    if not os.path.exists(args.output_label):
        os.makedirs(args.output_label)

    img_labels = pd.read_csv(args.file_input)

    test_file_headers = ["FileName"]
    test_file_keyword_headers = ["Diagnostic Keywords"]
    test_keywords = ["glaucoma", "atrophy", "valid_image"]
    glaucoma_transfer_header = "Keyword Values"

    #test_file_headers = ["Left-Fundus", "Right-Fundus"]
    #test_file_keyword_headers = ["Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]
    #test_keywords = ["glaucoma", "atrophy", "valid_image"]

    for index, row in img_labels.iterrows():
        for i in range(len(test_file_headers)):
            filename = row[test_file_headers[i]]
            if os.path.isfile(os.path.join(args.image_input, filename)):
                row_keywords = row[test_file_keyword_headers[i]]
                result_values = [float(row_keywords.lower().find(keyword.lower()) != -1) for keyword in test_keywords]

                if result_values[0] == 1 or result_values[1] == 1:
                    with (open(os.path.join(args.image_input, filename), 'rb') as src,
                          open(os.path.join(args.output_image, filename), 'wb') as dst):
                        # Read the source file and write it to the destination
                        dst.write(src.read())

                if result_values[0] == 1:
                    result_values[0] = float(row[glaucoma_transfer_header])

                data = pd.DataFrame([result_values], columns=test_keywords)
                data.to_csv(f"{args.output_label}/{filename}.csv", index=False)



if __name__ == '__main__':
    main()
