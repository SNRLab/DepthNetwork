import logging
import os.path
import sys

import keras.backend
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from keras.optimizers import Adam

import depth_network.model as model


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

keras.backend.set_image_data_format('channels_first')

data_dir = 'data'
rgb_data_file = os.path.join(data_dir, 'rgb.hdf5')
rgb_train_data_file = os.path.join(data_dir, 'rgb_train.hdf5')
rgb_validation_data_file = os.path.join(data_dir, 'rgb_validation.hdf5')
brdf_data_file = os.path.join(data_dir, 'brdf.hdf5')
brdf_train_data_file = os.path.join(data_dir, 'brdf_train.hdf5')
brdf_validation_data_file = os.path.join(data_dir, 'brdf_validation.hdf5')
depth_data_file = os.path.join(data_dir, 'depth.hdf5')
depth_train_data_file = os.path.join(data_dir, 'depth_train.hdf5')
depth_validation_data_file = os.path.join(data_dir, 'depth_validation.hdf5')

render_model_file = os.path.join(data_dir, 'render_model.hdf5')
render_model_checkpoint_file = os.path.join(data_dir, 'render_model_{epoch:02d}_{val_loss:.5f}.hdf5')
depth_model_file = os.path.join(data_dir, 'depth_model.hdf5')
depth_model_checkpoint_file = os.path.join(data_dir, 'depth_model_{epoch:02d}_{val_loss:.5f}.hdf5')

log_dir = 'logs'

image_size = (64, 64)


def load_models(create=False):
    return load_render_model(create), load_depth_model(create)


def load_render_model(create=False):
    return _load_model(render_model_file, output_channels=3, loss='mean_absolute_error', name='render_net',
                       create=create)


def load_depth_model(create=False):
    return _load_model(depth_model_file, output_channels=1, name='depth_net', create=create)


def _load_model(file, input_shape=None, output_channels=1, loss='mean_squared_error', name='depth_net', create=False):
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    m = model.DepthNetwork(input_shape=input_shape, output_channels=output_channels, name=name)
    _compile_model(m, loss=loss)
    try:
        logger.info("Loading model weights from %s...", file)
        m.load_weights(file)
    except (OSError, ValueError) as e:
        logger.warning("Could not load model weights from %s: %s", file, e)
        if not create:
            return None
    return m


def _compile_model(m, loss='mean_squared_error'):
    logger = logging.getLogger(__name__)
    logger.info("Compiling model...")
    m.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=[dice_coef, 'accuracy'])


def preprocess_batch(images):
    new_images = np.empty((images.shape[0], images.shape[1]) + image_size)

    for i, image in enumerate(images):
        new_image = skimage.transform.rescale(image.transpose((1, 2, 0)), (0.5, 0.5), preserve_range=True,
                                              mode='constant').transpose((2, 0, 1))
        left_padding = round((image_size[1] - new_image.shape[2]) / 2)
        right_padding = 64 - new_image.shape[2] - left_padding
        top_padding = round((image_size[0] - new_image.shape[1]) / 2)
        bottom_padding = 64 - new_image.shape[1] - top_padding
        padding = ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding))
        new_image = np.pad(new_image, padding, mode='reflect')
        new_images[i] = new_image

    return new_images


def preprocess_rgb_batch(images):
    images /= 255.
    return preprocess_batch(images)


def preprocess_depth_batch(depths):
    # Rows and columns are switched in HDF files
    depths = np.transpose(depths, (0, 1, 3, 2))
    # Scale data between 0 and 1
    depths /= 150
    np.clip(depths, 0, 1, depths)
    return preprocess_batch(depths)


def preprocess_brdf_batch(brdfs):
    # Magic number to scale data into appropriate range
    brdfs /= 5000
    np.clip(brdfs, 0, 1, brdfs)
    return preprocess_batch(brdfs)


def render_data_normalizer(images, brdfs):
    images, brdfs = preprocess_rgb_batch(images), preprocess_brdf_batch(brdfs)
    return images, brdfs


def depth_data_normalizer(brdfs, depths):
    return preprocess_brdf_batch(brdfs), preprocess_depth_batch(depths)


def postprocess_rgb_batch(images):
    images = np.clip(images, 0, 1)
    return images[:, :3, 7:57, 7:57]


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
