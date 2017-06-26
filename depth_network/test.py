import h5py
import matplotlib.pyplot as plt
import numpy as np

import depth_network.common as common


def main():
    render_model, depth_model = common.load_models()

    rgb_data = h5py.File(common.rgb_data_file, 'r')
    depth_data = h5py.File(common.depth_data_file, 'r')

    rgb_images = rgb_data['.']['RGB']
    depth_images = depth_data['.']['Z']

    while True:
        try:
            i = int(input("Image Index: "))
        except(IndexError, ValueError):
            continue
        if i >= len(rgb_images):
            continue

        rgb_image = common.preprocess_rgb_batch(rgb_images[i:i + 1])
        depth_image = common.preprocess_depth_batch(depth_images[i:i + 1])

        depth_image_cl = common.postprocess_depth_batch(depth_image)[0][0]
        rgb_image_cl = np.transpose(common.postprocess_rgb_batch(rgb_image)[0], (1, 2, 0))
        output = depth_model.predict(rgb_image)
        common.show_images((rgb_image_cl, depth_image_cl, common.postprocess_depth_batch(output)[0][0]))
        plt.show()


if __name__ == "__main__":
    main()
