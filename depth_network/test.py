import h5py
import matplotlib.pyplot as plt
import numpy as np

import depth_network.common as common


def main():
    model = common.load_model()

    rgb_data = h5py.File(common.rgb_data_file, 'r')
    depth_data = h5py.File(common.depth_data_file, 'r')

    rgb_images = rgb_data['.']['RGB']
    depth_images = depth_data['.']['Z']

    while True:
        i = int(input("Image Index: "))

        rgb_image = common.preprocess_rgb(rgb_images[i:i + 1])
        depth_image = common.preprocess_depth(depth_images[i:i + 1])

        depth_image_cl = common.postprocess_depth(depth_image)[0][0]
        rgb_image_cl = np.transpose(common.postprocess_rgb(rgb_image)[0], (1, 2, 0))
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        fig.set_tight_layout(True)
        axes.ravel()[0].imshow(rgb_image_cl)
        axes.ravel()[1].imshow(depth_image_cl)
        output = model.predict(rgb_image)
        axes.ravel()[2].imshow(common.postprocess_depth(output)[0][0])
        plt.show()


if __name__ == "__main__":
    main()
