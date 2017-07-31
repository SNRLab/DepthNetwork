"""
Implementation of the neural network as well as model specific utility functions
for pre/postprocessing data
"""

import keras.backend as K
import numpy as np
import skimage.transform
from keras.layers import MaxPooling2D, UpSampling2D, Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from depth_network.common import image_size, depth_divider, brdf_divider

K.set_image_data_format('channels_first')


def DepthNetwork(input_shape=None, output_channels=1, data_format=None, name='depth_net'):
    """
    Create an instance of the network. Both the render and depth networks have
    the same structure and are instantiated using this function.

    Usually, the helpers functions in `common` are used instead of this function.

    :param input_shape: the shape of the input tensor
    :param output_channels: number of channels in the output tensor
    :param data_format: 'channels_first' or 'channels_last'
    :param name: name of the network
    :return: an instance of the model
    """
    if data_format is None:
        data_format = K.image_data_format()

    if input_shape is None:
        if data_format == 'channels_first':
            input_shape = (3, 64, 64)
        else:
            input_shape = (64, 64, 3)

    inputs = Input(input_shape)
    x = _alpha(inputs, 16, "alpha1", data_format=data_format)
    x = MaxPooling2D(2, data_format=data_format)(x)
    x = _alpha(x, 32, "alpha2", data_format=data_format)
    x = MaxPooling2D(2, data_format=data_format)(x)
    x = _alpha(x, 64, "alpha3", data_format=data_format)
    x = MaxPooling2D(2, data_format=data_format)(x)
    x = _alpha(x, 128, "alpha4", data_format=data_format)
    x = MaxPooling2D(2, data_format=data_format)(x)
    x = _beta(x, 256, "beta_bottleneck", data_format=data_format)
    x = UpSampling2D(2, data_format=data_format)(x)
    x = _alpha_transpose(x, 128, "alpha5", data_format=data_format)
    x = UpSampling2D(2, data_format=data_format)(x)
    x = _alpha_transpose(x, 64, "alpha6", data_format=data_format)
    x = UpSampling2D(2, data_format=data_format)(x)
    x = _alpha_transpose(x, 32, "alpha7", data_format=data_format)
    x = UpSampling2D(2, data_format=data_format)(x)
    x = _alpha_transpose(x, 16, "alpha8", data_format=data_format)
    x = Conv2DTranspose(filters=output_channels, kernel_size=3, padding='same', name='output_conv',
                        data_format=data_format)(x)
    x = Activation('relu')(x)

    return Model(inputs=inputs, outputs=x, name=name)


def _alpha(input_tensor, filters, name, data_format=None):
    x = _beta(input_tensor, filters, name + "_beta1", data_format=data_format)
    x = _beta(x, filters, name + "_beta2", data_format=data_format)
    x = _beta(x, filters, name + "_beta3", data_format=data_format)
    return x


def _alpha_transpose(input_tensor, filters, name, data_format=None):
    x = _beta_transpose(input_tensor, filters, name + "_beta1", data_format=data_format)
    x = _beta_transpose(x, filters, name + "_beta2", data_format=data_format)
    x = _beta_transpose(x, filters, name + "_beta3", data_format=data_format)
    return x


def _beta(input_tensor, filters, name, data_format=None):
    x = Conv2D(filters=filters, kernel_size=3, padding='same', name=name + "_conv",
               data_format=data_format)(input_tensor)
    return _beta_common(x, name, data_format=data_format)


def _beta_transpose(input_tensor, filters, name, data_format=None):
    x = Conv2DTranspose(filters=filters, kernel_size=3, padding='same', name=name + "_conv_transpose",
                        data_format=data_format)(input_tensor)
    x = _beta_common(x, name, data_format=data_format)
    return x


def _beta_common(input_tensor, name, data_format=None):
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = 3
    x = BatchNormalization(axis=axis, name=name + "_batch_norm")(input_tensor)
    x = Activation('relu', name=name + "_activation")(x)
    return x


def preprocess_batch(images):
    """
    Process a batch of images to feed to the network. The images should be
    100x100 in channels last format.

    :param images: images to process
    :return: the processed batch
    """
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
    """
    Process a batch of RGB images to feed to the network. The images should be
    100x100 in channels last format, with a range of 0-255.

    :param images: images to process
    :return: the processed batch
    """
    images /= 255.
    return preprocess_batch(images)


def preprocess_depth_batch(depths):
    # Rows and columns are switched in HDF files
    depths = np.transpose(depths, (0, 1, 3, 2))
    # Scale data between 0 and 1
    depths /= depth_divider
    np.clip(depths, 0, 1, depths)
    return preprocess_batch(depths)


def preprocess_brdf_batch(brdfs):
    # Magic number to scale data into appropriate range
    brdfs /= brdf_divider
    np.clip(brdfs, 0, 1, brdfs)
    return preprocess_batch(brdfs)


def postprocess_rgb_batch(images):
    images = np.clip(images, 0, 1)
    return images[:, :3, 7:57, 7:57]


def postprocess_depth_batch(depth):
    return depth[:, :1, 7:57, 7:57]
