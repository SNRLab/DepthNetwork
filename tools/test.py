#!/usr/bin/env python3

import argparse
import logging
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="View outputs from trained models.")
    parser.add_argument('-r', '--render-model', help="render model weights file")
    parser.add_argument('-d', '--depth-model', help="depth model weights file")
    parser.add_argument('--rgb', help='RGB data file')
    parser.add_argument('--brdf', help='BRDF data file', required=True)
    parser.add_argument('--depth', help='depth data file')
    args = parser.parse_args()

    if args.render_model is None and args.depth_model is None:
        parser.error("at least one of --depth-model and --render-model must be specified")
    if args.render_model is not None and args.rgb is None:
        parser.error("--render-model requires --rgb")
    if args.depth_model is not None and args.depth is None:
        parser.error("--depth-model requires --depth")

    # Keras takes a long time to load, so defer it until after argument parsing
    import depth_network.common as common

    if args.render_model is not None:
        render_model = common.load_render_model(args.render_model, create=False)
        if render_model is None:
            logger.critical("Could not load render model.")
            sys.exit(1)
    else:
        render_model = None

    if args.depth_model is not None:
        depth_model = common.load_depth_model(args.depth_model, create=False)
        if depth_model is None:
            logger.critical("Could not load depth model.")
            sys.exit(1)
    else:
        depth_model = None

    if args.rgb:
        rgb_data = h5py.File(args.rgb, 'r')
        rgb_images = rgb_data['RGB']
    else:
        rgb_images = None

    brdf_data = h5py.File(args.brdf, 'r')
    brdf_images = brdf_data['BRDF']

    if args.depth:
        depth_data = h5py.File(args.depth, 'r')
        depth_images = depth_data['Z']
    else:
        depth_images = None

    if rgb_images is not None and len(brdf_images) != len(rgb_images):
        logger.critical("Length of RGB data (%d) and BRDF data (%d) does not match", len(rgb_images), len(brdf_images))
        sys.exit(3)

    if depth_images is not None and len(brdf_images) != len(depth_images):
        logger.critical("Length of depth data (%d) and BRDF data (%d) does not match", len(depth_images),
                        len(brdf_images))
        sys.exit(3)

    while True:
        try:
            i = int(input("Image Index: "))
        except(IndexError, ValueError):
            logger.error("Invalid index format")
            continue
        if i >= len(brdf_images):
            logger.error("Index out of range")
            continue

        images = []

        if rgb_images is not None:
            # Preprocess RGB ground truth image
            rgb_batch_true_pre = common.preprocess_rgb_batch(rgb_images[i:i + 1])
            rgb_image_true = np.transpose(common.postprocess_rgb_batch(rgb_batch_true_pre)[0], (1, 2, 0)).astype(
                np.float32)
            rgb_image_true = cv2.cvtColor(rgb_image_true, cv2.COLOR_BGR2RGB)
            images.append(rgb_image_true)

        # Preprocess BRDF ground truth image
        brdf_batch_true_pre = common.preprocess_brdf_batch(brdf_images[i:i + 1])
        brdf_image_true = np.transpose(common.postprocess_rgb_batch(brdf_batch_true_pre)[0], (1, 2, 0)).astype(
            np.float32)
        brdf_image_true = cv2.cvtColor(brdf_image_true, cv2.COLOR_BGR2RGB)
        images.append(brdf_image_true)

        if depth_images is not None:
            # Preprocess depth ground truth image
            depth_batch_true_pre = common.preprocess_depth_batch(depth_images[i:i + 1])
            depth_image_true = common.postprocess_depth_batch(depth_batch_true_pre)[0][0]
            images.append(depth_image_true)

        if render_model is not None:
            # Predict BRDF from ground truth RGB image
            brdf_batch_pred = render_model.predict(rgb_batch_true_pre)
            brdf_batch_pred_post = common.postprocess_rgb_batch(brdf_batch_pred)

            brdf_image_pred = np.transpose(brdf_batch_pred_post[0], (1, 2, 0))
            brdf_image_pred = cv2.cvtColor(brdf_image_pred, cv2.COLOR_BGR2RGB)
            brdf_diff = np.abs(brdf_image_true - brdf_image_pred)
            logger.info('BRDF: mse: %s, mae: %s', (brdf_diff ** 2).mean(), brdf_diff.mean())
            images.append(brdf_image_pred)

        if depth_model is not None:
            if render_model is not None:
                # Predict depth from predicted BRDF
                depth_batch_pred_post = common.postprocess_depth_batch(depth_model.predict(brdf_batch_pred))
                depth_image_pred = depth_batch_pred_post[0][0]
                images.append(depth_image_pred)

            # Predict depth from ground truth BRDF
            depth_batch_pred2_post = common.postprocess_depth_batch(depth_model.predict(brdf_batch_true_pre))
            depth_image_pred2 = depth_batch_pred2_post[0][0]
            images.append(depth_image_pred2)

        common.show_images(images)
        plt.show()


if __name__ == "__main__":
    main()
