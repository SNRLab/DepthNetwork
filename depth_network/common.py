import os.path

import keras.backend
import keras.models
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import depth_network.model as model
import logging

keras.backend.set_image_data_format('channels_first')

data_dir = 'data'
rgb_data_file = os.path.join(data_dir, 'rgb.hdf5')
brdf_data_file = os.path.join(data_dir, 'brdf.hdf5')
depth_data_file = os.path.join(data_dir, 'depth.hdf5')

render_model_file = os.path.join(data_dir, 'render_model.hdf5')
depth_model_file = os.path.join(data_dir, 'depth_model.hdf5')

log_dir = 'logs'

image_size = (64, 64)


def load_models(create=False):
    return load_render_model(create), load_depth_model(create)


def load_render_model(create=False):
    return _load_model(render_model_file, create)


def load_depth_model(create=False):
    return _load_model(depth_model_file, create)


def _load_model(file, create=False):
    m = model.DepthNetwork(input_shape=(3,) + image_size)
    m.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=[dice_coef, 'accuracy'])
    try:
        m.load_weights(file)
    except (OSError, ValueError) as e:
        logging.warning("Could not load network from %s: %s", file, e)
        if not create:
            return None
    return m


def preprocess_batch(images, channel_padding):
    new_images = np.empty((images.shape[0], images.shape[1] + channel_padding,) + image_size)

    for i, image in enumerate(images):
        new_image = skimage.transform.rescale(image.transpose((1, 2, 0)), (0.5, 0.5), preserve_range=True,
                                              mode='constant').transpose((2, 0, 1))
        left_padding = round((image_size[1] - new_image.shape[2]) / 2)
        right_padding = 64 - new_image.shape[2] - left_padding
        top_padding = round((image_size[0] - new_image.shape[1]) / 2)
        bottom_padding = 64 - new_image.shape[1] - top_padding
        padding = ((0, channel_padding), (top_padding, bottom_padding), (left_padding, right_padding))
        new_image = np.pad(new_image, padding, mode='reflect')
        new_images[i] = new_image
    return new_images


def preprocess_rgb_batch(images):
    images /= 255.
    return preprocess_batch(images, 0)


def preprocess_brdf_output_batch(brdfs):
    return preprocess_batch(brdfs, 16 - brdfs.shape[1])


def preprocess_brdf_input_batch(brdfs):
    return preprocess_batch(brdfs, 0)


def preprocess_depth_batch(depths):
    depths = depths.astype(np.uint8)
    depths = np.transpose(depths, (0, 1, 3, 2))
    return preprocess_batch(depths, 16 - depths.shape[1])


def render_data_normalizer(images, brdfs):
    return preprocess_rgb_batch(images), preprocess_brdf_output_batch(brdfs)


def depth_data_normalizer(brdfs, depths):
    return preprocess_brdf_input_batch(brdfs), preprocess_depth_batch(depths)


def postprocess_rgb_batch(image):
    return image[:, :3, 7:57, 7:57]


def postprocess_depth_batch(depth):
    return depth[:, :1, 7:57, 7:57]


def normalize_image_range(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def show_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    fig.set_tight_layout(True)
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img)
    plt.show()


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
