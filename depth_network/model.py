import keras.backend as K
from keras.models import Model
from keras.layers import MaxPooling2D, UpSampling2D, Activation, Input, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization


def DepthNetwork(input_shape=None):
    if input_shape is None:
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 64, 64)
        else:
            input_shape = (64, 64, 3)

    inputs = Input(input_shape)
    # if K.image_data_format() == 'channels_first':
    #     new_shape = (input_shape[1], input_shape[0] * input_shape[2])
    # else:
    #     new_shape = (input_shape[0], input_shape[2] * input_shape[1])
    # x = Reshape(new_shape)(inputs)
    x = _alpha(inputs, 16, "alpha1")
    x = MaxPooling2D(2)(x)
    x = _alpha(x, 32, "alpha2")
    x = MaxPooling2D(2)(x)
    x = _alpha(x, 64, "alpha3")
    x = MaxPooling2D(2)(x)
    x = _alpha(x, 128, "alpha4")
    x = MaxPooling2D(2)(x)
    x = _beta(x, 256, "beta_bottleneck")
    x = UpSampling2D(2)(x)
    x = _alpha_transpose(x, 128, "alpha5")
    x = UpSampling2D(2)(x)
    x = _alpha_transpose(x, 64, "alpha6")
    x = UpSampling2D(2)(x)
    x = _alpha_transpose(x, 32, "alpha7")
    x = UpSampling2D(2)(x)
    x = _alpha_transpose(x, 16, "alpha8")

    return Model(inputs=inputs, outputs=x)


def _alpha(input_tensor, filters, name):
    x = _beta(input_tensor, filters, name + "_beta1")
    x = _beta(x, filters, name + "_beta2")
    x = _beta(x, filters, name + "_beta3")
    return x


def _alpha_transpose(input_tensor, filters, name):
    x = _beta_transpose(input_tensor, filters, name + "_beta1")
    x = _beta_transpose(x, filters, name + "_beta2")
    x = _beta_transpose(x, filters, name + "_beta3")
    return x


def _beta(input_tensor, filters, name):
    x = Conv2D(filters=filters, kernel_size=3, padding='same', name=name + "_conv")(input_tensor)
    return _beta_common(x, name)


def _beta_transpose(input_tensor, filters, name):
    x = Conv2DTranspose(filters=filters, kernel_size=3, padding='same', name=name + "_conv_transpose")(input_tensor)
    x = _beta_common(x, name)
    return x


def _beta_common(input_tensor, name):
    if K.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3
    x = BatchNormalization(axis=axis, name=name + "_batch_norm")(input_tensor)
    x = Activation('relu', name=name + "_activation")(x)
    return x

