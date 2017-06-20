import keras.backend as K
from depth_network.better_io_utils import BetterHDF5Matrix
from depth_network.model import DepthNetwork
import numpy as np

K.set_image_data_format('channels_first')

image_size = (64, 64)


def image_normalizer(image):
    print(image.shape)
    return np.pad(np.resize(image, (image.shape[0], 50, 50)), 64 - 50, mode='reflect')


def depth_normalizer(depth):
    padding = ((0, 15), (64 - 50, 64 - 50), (64 - 50, 64 - 50))
    return np.pad(np.resize(depth, (depth.shape[0], 50, 50)), padding, mode='reflect')


def main():
    rgb_data = BetterHDF5Matrix('data/0_rgb_small.hdf5', 'RGB', normalizer=image_normalizer,
                                normalized_shape=(3,) + image_size)
    depth_data = BetterHDF5Matrix('data/0_z_small.hdf5', 'Z', normalizer=depth_normalizer,
                                  normalized_shape=(16,) + image_size)

    model = DepthNetwork(input_shape=(3, 64, 64))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(rgb_data, depth_data, batch_size=1, epochs=1)


if __name__ == "__main__":
    main()
