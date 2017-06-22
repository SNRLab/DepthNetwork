import depth_network.common as common
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
        print(rgb_image[0].shape)
        print(depth_image[0].shape)

        depth_image_cl = depth_image[0][0]
        rgb_image_cl = np.transpose(rgb_image[0], (1, 2, 0))
        print(depth_image_cl.shape)
        print(rgb_image_cl.shape)
        plt.figure()
        plt.imshow(rgb_image_cl)
        plt.figure()
        plt.imshow(depth_image_cl)
        plt.figure()
        output = model.predict(rgb_image)[0][0]
        # output_cl = np.transpose(output, (1, 2, 0))
        plt.imshow(output)
        plt.show()


if __name__ == "__main__":
    main()
