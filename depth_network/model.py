import keras.backend as K
from keras.layers import MaxPooling2D, UpSampling2D, Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def DepthNetwork(input_shape=None, output_channels=1, data_format=None, name='depth_net'):
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
