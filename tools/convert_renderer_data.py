#!/usr/bin/env python3

import argparse
import glob
import logging
import os.path
import sys

import cv2
import h5py
import numpy as np

import depth_network.data_utils as data_utils
import depth_network.common as common

IMAGE_SIZE = (100, 100)

logging.basicConfig(level=logging.INFO)

_mask = np.zeros(IMAGE_SIZE, np.uint8)
cv2.circle(_mask, (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), IMAGE_SIZE[0] // 2, 1, -1)


def main():
    logger = logging.getLogger('convert_unity_data')

    parser = argparse.ArgumentParser(description=
                                     "Converts PNG and EXR data files created by the Unity renderer to HDF5 datasets. "
                                     "This script makes some naive assumptions, such as that the files have not been "
                                     "renamed and that all the files in 'data_dir' are generated by Unity. After "
                                     "creating the HDF5 file, the input files should be deleted, so they don't get "
                                     "accidentally converted again in the future.")
    parser.add_argument('--brdf-output', help='BRDF data output file', required=True)
    parser.add_argument('--depth-output', help='Depth data output file', required=True)
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    brdf_file = h5py.File(args.brdf_output, 'w')
    depth_file = h5py.File(args.depth_output, 'w')

    brdf_images = glob.glob(os.path.join(args.data_dir, 'brdf_*.png'))
    depth_images = glob.glob(os.path.join(args.data_dir, 'depth_*.exr'))

    if len(brdf_images) != len(depth_images):
        logger.critical("Unequal numbers of BRDF and depth images")
        sys.exit(1)

    brdf_dataset = brdf_file.create_dataset('BRDF', (len(brdf_images), 3) + IMAGE_SIZE, np.float32)
    depth_dataset = depth_file.create_dataset('Z', (len(brdf_images), 1) + IMAGE_SIZE, np.float16)

    for i, (brdf_image, depth_image) in enumerate(zip(sorted(brdf_images), sorted(depth_images))):
        # Not super robust or flexible, but it will work for now
        if brdf_image[-28:-4] != depth_image[-28:-4]:
            logger.critical("Depth image (%s) and BRDF image (%s) not paired correctly", brdf_image, depth_image)
            sys.exit(2)

        # The BRDF data included in the paper has a very large magnitude, so we scale it down between 0 and 1. My
        # renderer outputs values between 0 and 255, so we have to scale them to have the same range as the data from
        # the paper
        brdf_image = (cv2.imread(brdf_image) / 255) * common.brdf_divider
        brdf_image[_mask == 0] = 0
        brdf_image = np.rollaxis(brdf_image, 2, 0)
        depth_image = data_utils.read_exr_depth(depth_image).copy()
        depth_image[_mask == 0] = 0
        depth_image = np.rollaxis(depth_image, 2, 0)

        assert brdf_image.shape == (3,) + IMAGE_SIZE
        assert depth_image.shape == (1,) + IMAGE_SIZE

        brdf_dataset[i] = brdf_image
        depth_dataset[i] = depth_image


if __name__ == '__main__':
    main()
