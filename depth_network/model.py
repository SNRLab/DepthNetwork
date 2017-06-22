import keras.backend.common
import keras.layers
import keras.models
from keras.layers import MaxPooling2D, UpSampling2D, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization


def DepthNetwork(input_shape=(64, 64, 3)):
    model = keras.models.Sequential()
    _add_alpha(model, 16, "alpha1", input_shape=input_shape)
    model.add(MaxPooling2D(2))
    _add_alpha(model, 32, "alpha2")
    model.add(MaxPooling2D(2))
    _add_alpha(model, 64, "alpha3")
    model.add(MaxPooling2D(2))
    _add_alpha(model, 128, "alpha4")
    model.add(MaxPooling2D(2))
    _add_beta(model, 256, "beta_bottleneck")
    model.add(UpSampling2D(2))
    _add_alpha_transpose(model, 128)
    model.add(UpSampling2D(2))
    _add_alpha_transpose(model, 64)
    model.add(UpSampling2D(2))
    _add_alpha_transpose(model, 32)
    model.add(UpSampling2D(2))
    _add_alpha_transpose(model, 16)

    return model


def _add_alpha(model, filters, name, input_shape=None):
    _add_beta(model, filters, name + "_beta1", input_shape)
    _add_beta(model, filters, name + "_beta2", input_shape)
    _add_beta(model, filters, name + "_beta3", input_shape)


def _add_alpha_transpose(model, filters):
    _add_beta_transpose(model, filters)
    _add_beta_transpose(model, filters)
    _add_beta_transpose(model, filters)


def _add_beta(model, filters, name, input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters=filters, kernel_size=3, padding='same', name=name + "_conv",
                         input_shape=input_shape))
    else:
        model.add(Conv2D(filters=filters, kernel_size=3, padding='same', name=name + "_conv"))
    model.add(BatchNormalization(axis=1))  # axis = 3 for channels_last
    model.add(Activation('relu'))


def _add_beta_transpose(model, filters):
    model.add(Conv2DTranspose(filters=filters, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=1))  # axis = 3 for channels_last
    model.add(Activation('relu'))
