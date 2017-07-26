import logging
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

import depth_network.common as common

_logger = logging.getLogger(__name__)


def main():
    render_model, depth_model = common.load_models(create=True)
    if render_model is None or depth_model is None:
        _logger.critical("Could not load render and/or depth models.")
        sys.exit(1)

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
        brdf_batch_true_pre = common.preprocess_brdf_batch(brdf_images[i:i + 1])
        depth_batch_true_pre = common.preprocess_depth_batch(depth_images[i:i + 1])

        rgb_image_true = np.transpose(common.postprocess_rgb_batch(rgb_batch_true_pre)[0], (1, 2, 0)).astype(np.float32)
        rgb_image_true = cv2.cvtColor(rgb_image_true, cv2.COLOR_BGR2RGB)
        brdf_image_true = np.transpose(common.postprocess_rgb_batch(brdf_batch_true_pre)[0], (1, 2, 0)).astype(
            np.float32)
        brdf_image_true = cv2.cvtColor(brdf_image_true, cv2.COLOR_BGR2RGB)
        depth_image_true = common.postprocess_depth_batch(depth_batch_true_pre)[0][0]

        brdf_batch_pred = render_model.predict(rgb_batch_true_pre)
        brdf_batch_pred_post = common.postprocess_rgb_batch(brdf_batch_pred)
        depth_batch_pred_post = common.postprocess_depth_batch(depth_model.predict(brdf_batch_pred))

        depth_batch_pred2_post = common.postprocess_depth_batch(depth_model.predict(brdf_batch_true_pre))

        brdf_image_pred = np.transpose(brdf_batch_pred_post[0], (1, 2, 0))
        brdf_image_pred = cv2.cvtColor(brdf_image_pred, cv2.COLOR_BGR2RGB)
        brdf_diff = np.abs(brdf_image_true - brdf_image_pred)
        _logger.info('index: %s, mse: %s, mae: %s', i, (brdf_diff ** 2).mean(), brdf_diff.mean())

        depth_image_pred = depth_batch_pred_post[0][0]
        depth_image_pred2 = depth_batch_pred2_post[0][0]
        common.show_images((rgb_image_true, brdf_image_true, depth_image_true, brdf_image_pred, depth_image_pred,
                            depth_image_pred2))
        plt.show()


if __name__ == "__main__":
    main()
