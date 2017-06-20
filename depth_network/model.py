import keras.models
from keras.layers import MaxPooling2D, UpSampling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D


def DepthNetwork(input_shape=(64, 64, 3)):
    model = keras.models.Sequential()
    _add_alpha(model, 16, input_shape=input_shape)
    model.add(MaxPooling2D(2))
    _add_alpha(model, 32)
    model.add(MaxPooling2D(2))
    _add_alpha(model, 64)
    model.add(MaxPooling2D(2))
    _add_alpha(model, 128)
    model.add(MaxPooling2D(2))
    _add_beta(model, 256)
    model.add(UpSampling2D(2))
    _add_alpha(model, 128)
    model.add(UpSampling2D(2))
    _add_alpha(model, 64)
    model.add(UpSampling2D(2))
    _add_alpha(model, 32)
    model.add(UpSampling2D(2))
    _add_alpha(model, 16)

    return model


def _add_alpha(model, filters, input_shape=None):
    _add_beta(model, filters, input_shape)
    _add_beta(model, filters, input_shape)
    _add_beta(model, filters, input_shape)


def _add_beta(model, filters, input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters=filters, kernel_size=3, padding='same', input_shape=input_shape))
    else:
        model.add(Conv2D(filters=filters, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=1))  # axis = 3 for channels_last
    model.add(Activation('relu'))
