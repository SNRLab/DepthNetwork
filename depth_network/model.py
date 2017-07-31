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


def preprocess_batch(images, data_format=None):
    """
    Process a batch of images to feed to the network. The images should be
    square.

    Normally one of the format specific preprocessing functions should be
    called rather than this one.

    This function assumes is not completely configurable, as it always resizes
    images to 50x50 and pads them to 64x64 as was done in the paper.

    :param images: images to process
    :param data_format: 'channels_last' or 'channels_first'
    :return: the processed batch
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_first', 'channels_last')

    # SciKit image always uses channels last
    if data_format == 'channels_first':
        images = images.transpose((0, 2, 3, 1))
    new_images = np.empty(images.shape[:1] + image_size + images.shape[-1:])

    for i, image in enumerate(images):
        new_image = skimage.transform.resize(image, (50, 50), preserve_range=True, mode='constant')
        left_padding = round((image_size[1] - new_image.shape[1]) / 2)
        right_padding = 64 - new_image.shape[1] - left_padding
        top_padding = round((image_size[0] - new_image.shape[0]) / 2)
        bottom_padding = 64 - new_image.shape[0] - top_padding
        padding = ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0))
        new_image = np.pad(new_image, padding, mode='reflect')
        new_images[i] = new_image

    if data_format == 'channels_first':
        new_images = new_images.transpose((0, 3, 1, 2))

    return new_images


def preprocess_rgb_batch(images, data_format=None):
    """
    Process a batch of RGB images to feed to the network. The images should be
    square, with a range of 0-255.

    :param images: RGB images to process
    :param data_format: 'channels_last' or 'channels_first'
    :return: the processed batch
    """
    images /= 255.
    return preprocess_batch(images, data_format)


def preprocess_depth_batch(depths, swap_axes=False, data_format=None):
    """
    Process a batch of depth images to feed to the network. The images should
    be square and in millimeters.

    This function optionally swaps the x and y axes because the sample data
    from the paper has them swapped.

    :param depths: depth images to process
    :param swap_axes: if true, swap the x and y axes
    :param data_format: 'channels_last' or 'channels_first'
    :return: the processed batch
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_first', 'channels_last')

    if swap_axes:
        print('swapping axes')
        # Rows and columns are switched in HDF files from paper
        if data_format == 'channels_first':
            depths = np.transpose(depths, (0, 1, 3, 2))
        else:
            depths = np.transpose(depths, (0, 2, 1, 3))
    # Scale data between 0 and 1
    depths /= depth_divider
    np.clip(depths, 0, 1, depths)
    return preprocess_batch(depths, data_format)


def preprocess_brdf_batch(brdfs, data_format=None):
    """
    Process a batch of BRDF images to feed to the network. The images should
    be square.

    :param brdfs: BRDF images to process
    :param data_format: 'channels_last' or 'channels_first'
    :return: the processed batch
    """
    # Magic divider to scale data between 0 and 1
    brdfs /= brdf_divider
    np.clip(brdfs, 0, 1, brdfs)
    return preprocess_batch(brdfs, data_format)


def postprocess_rgb_batch(images, data_format=None):
    """
    Reverse the preprocessing of an RGB image. This does not exactly restore the
    original image, because some of the pre-processing steps are destructive.
    This can be used to process the input to the render network for display and
    debugging, as well as to post-process the output of the render network.

    :param images: images to post-process
    :param data_format: 'channels_first' or 'channels_last'
    :return: post-processed images
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_first', 'channels_last')

    images = np.clip(images, 0, 1)
    if data_format == 'channels_first':
        return images[:, :3, 7:57, 7:57]
    else:
        return images[:, 7:57, 7:57, :3]


def postprocess_depth_batch(depths, data_format=None):
    """
    Post-process the output of the depth network.

    :param depths: depth images to post-process
    :param data_format: 'channels_first' or 'channels_last'
    :return: post-processed depth images
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_first', 'channels_last')

    if data_format == 'channels_first':
        return depths[:, :1, 7:57, 7:57]
    else:
        return depths[:, 7:57, 7:57, :1]
