import logging
import sys

import matplotlib.pyplot as plt
import numpy as np


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

image_size = (64, 64)
depth_divider = 150
brdf_divider = 5000


def load_render_model(file, create=False):
    """
    Load a render model from a file or create a new one.

    :param file: name of the file, or `None`, in which case, `create` must be
    `True`
    :param create: if `False`, the model won't be created if the file cannot be
    loaded
    :return: the model, or `None` if it could not be created
    """
    return _load_model(file, output_channels=3, loss='mean_absolute_error', name='render_net', create=create)


def load_depth_model(file, create=False):
    """
    Load a depth model from a file or create a new one.

    :param file: name of the file, or `None`, in which case, `create` must be
    `True`
    :param create: if `False`, the model won't be created if the file cannot be
    loaded
    :return: the model, or `None` if it could not be created
    """
    return _load_model(file, output_channels=1, name='depth_net', create=create)


def _load_model(file, input_shape=None, output_channels=1, loss='mean_squared_error', name='depth_net', create=False):
    import depth_network.model

    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    m = depth_network.model.DepthNetwork(input_shape=input_shape, output_channels=output_channels, name=name)
    _compile_model(m, loss=loss)
    if file is not None:
        try:
            logger.info("Loading model weights from %s...", file)
            m.load_weights(file)
        except (OSError, ValueError) as e:
            logger.warning("Could not load model weights from %s: %s", file, e)
            if not create:
                return None
    elif not create:
        logger.error("Tried to force loading model from file, but no file was specified")
        return None
    return m


def _compile_model(m, loss='mean_squared_error'):
    from keras.optimizers import Adam

    logger = logging.getLogger(__name__)
    logger.info("Compiling model...")
    m.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=[dice_coef, 'accuracy'])


def render_data_normalizer(images, brdfs):
    from depth_network.model import preprocess_rgb_batch, preprocess_brdf_batch

    images, brdfs = preprocess_rgb_batch(images), preprocess_brdf_batch(brdfs)
    return images, brdfs


def depth_data_normalizer(brdfs, depths, swap_depth_axes=False):
    from depth_network.model import preprocess_depth_batch, preprocess_brdf_batch

    return preprocess_brdf_batch(brdfs), preprocess_depth_batch(depths, swap_axes=swap_depth_axes)


def normalize_image_range(image):
    """
    Normalizes an image between 1 and 0.

    :param image: image to normalize
    :return: normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def show_images(images, titles=None):
    """
    Show a list of images using matplotlib. Color images are assumed to be BGR.

    :param images: images to display
    :param titles: titles of the images
    """
    if titles is None:
        titles = ("",) * len(images)
    else:
        assert len(titles) == len(images)
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    fig.set_tight_layout(True)
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.set_title(title)
        if img.shape[-1] == 3:
            img = img[..., ::-1]
        ax.imshow(img)
    plt.show()


smooth = 1.


def dice_coef(y_true, y_pred):
    import keras.models

    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
