import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import logging

import depth_network.common as common


def main():
    render_model, depth_model = common.load_models()
    if render_model is None or depth_model is None:
        logging.critical("Could not load render and/or depth models.")
        sys.exit(1)

    render_model.summary()

    rgb_data = h5py.File(common.rgb_data_file, 'r')
    brdf_data = h5py.File(common.brdf_data_file, 'r')
    depth_data = h5py.File(common.depth_data_file, 'r')

    rgb_images = rgb_data['RGB']
    brdf_images = brdf_data['BRDF']
    depth_images = depth_data['Z']

    while True:
        try:
            i = int(input("Image Index: "))
        except(IndexError, ValueError):
            continue
        if i >= len(rgb_images):
            continue

        rgb_batch_true_pre = common.preprocess_rgb_batch(rgb_images[i:i + 1])
        brdf_batch_true_pre = common.preprocess_brdf_output_batch(brdf_images[i:i + 1])
        depth_batch_true_pre = common.preprocess_depth_batch(depth_images[i:i + 1])

        rgb_image_true = np.transpose(common.postprocess_rgb_batch(rgb_batch_true_pre)[0], (1, 2, 0)).astype(np.float32)
        rgb_image_true = cv2.cvtColor(rgb_image_true, cv2.COLOR_BGR2RGB)
        brdf_image_true = common.normalize_image_range(
            np.transpose(common.postprocess_rgb_batch(brdf_batch_true_pre)[0], (1, 2, 0))).astype(np.float32)
        brdf_image_true = cv2.cvtColor(brdf_image_true, cv2.COLOR_BGR2RGB)
        depth_image_true = depth_batch_true_pre[0][0]

        brdf_batch_pred_post = common.postprocess_rgb_batch(render_model.predict(rgb_batch_true_pre))
        brdf_batch_pred_pre = common.preprocess_brdf_input_batch(brdf_batch_pred_post)
        depth_batch_pred_post = common.postprocess_depth_batch(depth_model.predict(brdf_batch_pred_pre))

        brdf_image_pred = common.normalize_image_range(np.transpose(brdf_batch_pred_post[0], (1, 2, 0)))
        print(brdf_image_pred.shape)
        depth_image_pred = depth_batch_pred_post[0][0]
        common.show_images((rgb_image_true, brdf_image_true, depth_image_true, brdf_image_pred, depth_image_pred))
        plt.show()


if __name__ == "__main__":
    main()
