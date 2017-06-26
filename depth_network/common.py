import os.path

import keras.backend
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

np.set_printoptions(threshold=np.nan)

keras.backend.set_image_data_format('channels_first')

data_dir = 'data'
rgb_data_file = os.path.join(data_dir, 'rgb.hdf5')
depth_data_file = os.path.join(data_dir, 'depth.hdf5')

model_file = os.path.join(data_dir, 'model.hdf5')

log_dir = 'logs'

image_size = (64, 64)


def load_model():
    return keras.models.load_model(model_file, custom_objects={'dice_coef': dice_coef})


def preprocess_image(images, channel_padding):
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


def preprocess_rgb(image):
    image /= 255.
    return preprocess_image(image, 0)


def preprocess_depth(depth):
    depth = depth.astype(np.uint8)
    depth = np.transpose(depth, (0, 1, 3, 2))
    return preprocess_image(depth, 16 - depth.shape[1])


def data_normalizer(images, depths):
    return preprocess_rgb(images), preprocess_depth(depths)


def postprocess_rgb(image):
    return image[:, :3, 7:57, 7:57]


def postprocess_depth(depth):
    return depth[:, :1, 7:57, 7:57]


def show_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    fig.set_tight_layout(True)
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img)
    # plt.show()

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
