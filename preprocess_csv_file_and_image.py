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

parser = argparse.ArgumentParser(description='get out csv values from csv files and images into per-image file')
parser.add_argument("--file-input", type=str,
                    #default="training-data/preprocess-output/full_df.csv",
                    default="training-data/preprocess-output/local_hand_df.csv",
                    help="path to the csv dataset file")

parser.add_argument("--image-input", type=str,
                    default="training-data/retina-stuff-classifier/for-display/x64/nerves_defined_output",
                    help="path to the image files")

parser.add_argument("--output-image", type=str,
                    default="training-data/retina-stuff-classifier/for-display/x64/nerves_defined_output_glaucoma-atrophy",
                    help="path to the output dir for images with atrophy and glaucoma")

parser.add_argument("--output-label", type=str,
                    default="training-data/retina-stuff-classifier/for-display/x64/nerve_defined_labels",
                    help="path to the output directory for masks")

'''
so, this file compares filenames from csv file 
and puts images into approppriate directories if file names match
also create appropriate csv files per image file 
these csv file contain appropriate labels of 0/1 for diagnoses
'''


def main():

    args = parser.parse_args()

    if not os.path.exists(args.output_label):
        os.makedirs(args.output_label)

    img_labels = pd.read_csv(args.file_input)

    #local files
    test_file_headers = ["FileName"]
    test_file_keyword_headers = ["Diagnostic Keywords"]
    test_keywords = ["glaucoma", "atrophy", "valid_image"]
    glaucoma_transfer_header = "Keyword Values"

    #kaggle images
    '''test_file_headers = ["Left-Fundus", "Right-Fundus"]
    test_file_keyword_headers = ["Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]
    test_keywords = ["glaucoma", "atrophy", "valid_image"]'''

    for index, row in img_labels.iterrows():
        for i in range(len(test_file_headers)):
            filename = row[test_file_headers[i]]
            print(f"input file at: {filename}")
            row_keywords = row[test_file_keyword_headers[i]]
            result_values = [float(row_keywords.lower().find(keyword.lower()) != -1) for keyword in test_keywords]

            for file_i in range(0, 10):
                name, ext = os.path.splitext(filename)
                derived_name = name + "_" + str(file_i) + ext
                if os.path.isfile(os.path.join(args.image_input, derived_name)):
                    # select only glaucoma or atrophy into new folder
                    if result_values[0] == 1 or result_values[1] == 1:
                        with (open(os.path.join(args.image_input, derived_name), 'rb') as src,
                              open(os.path.join(args.output_image, derived_name), 'wb') as dst):
                            # Read the source file and write it to the destination
                            dst.write(src.read())

                    # for local files
                    if result_values[0] == 1:
                        result_values[0] = float(row[glaucoma_transfer_header])

                    data = pd.DataFrame([result_values], columns=test_keywords)
                    data.to_csv(f"{args.output_label}/{derived_name}.csv", index=False)
                else:
                    break


if __name__ == '__main__':
    main()
