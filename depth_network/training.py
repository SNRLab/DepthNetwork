"""
Utility functions to train the network.
"""

import logging

import h5py
from keras.callbacks import ModelCheckpoint, Callback

import depth_network.common as common
from depth_network.data_utils import HDFGenerator


class LoggerCallback(Callback):
    """
    Callback that prints a nicely formatted log of a training session. Keras's
    default logging callback only writes to stdout and assumes it is connected
    to a TTY, which makes it ugly when redirected to a file.
    """

    def __init__(self, base_logger):
        """
        Creates a LoggerCallback that logs to a child of the specified logger.
        The child logger uses the name of the model.

        :param base_logger: parent of the callback's logger
        """
        super().__init__()
        self._base_logger = base_logger
        self._logger = None
        self._model = None
        self._epochs = 0
        self._steps = 0
        self._batch_print_mod = 1

    def _format_metrics(self, logs):
        metric_str = ""
        for k in self.params['metrics']:
            if k in logs:
                metric_str += ", {}: {:0.5f}".format(k, logs[k])
        return metric_str

    def set_params(self, params):
        super(LoggerCallback, self).set_params(params)
        self._epochs = params['epochs']
        self._steps = params['steps']
        self._batch_print_mod = max(round(self._steps / 20), 1)

    def set_model(self, model):
        super(LoggerCallback, self).set_model(model)
        self._logger = self._base_logger.getChild(model.name)

    def on_train_begin(self, logs=None):
        self._logger.info("┌Started training")

    def on_epoch_begin(self, epoch, logs=None):
        self._logger.info("├┬Started epoch %d/%d", epoch + 1, self._epochs)

    def on_batch_end(self, batch, logs=None):
        if batch % self._batch_print_mod == 0:
            self._logger.info("│├─Batch %d/%d%s", batch + 1, self._steps, self._format_metrics(logs))

    def on_epoch_end(self, epoch, logs=None):
        self._logger.info("│└Finished epoch %d/%d%s", epoch + 1, self._epochs, self._format_metrics(logs))

    def on_train_end(self, logs=None):
        self._logger.info("└Finished training")


def _train_network(model, x_train_file, y_train_file, x_validation_file, y_validation_file, x_dataset, y_dataset,
                   checkpoint_file_format, output_file, data_normalizer, epochs, verbose=1):
    x_train_data = h5py.File(x_train_file, 'r')
    y_train_data = h5py.File(y_train_file, 'r')
    x_validation_data = h5py.File(x_validation_file, 'r')
    y_validation_data = h5py.File(y_validation_file, 'r')
    train_data_generator = HDFGenerator(x_train_data[x_dataset], y_train_data[y_dataset], batch_size=16,
                                        normalizer=data_normalizer, shuffle=True)
    validation_data_generator = HDFGenerator(x_validation_data[x_dataset], y_validation_data[y_dataset], batch_size=64,
                                             normalizer=data_normalizer, shuffle=False)

    model_checkpoint = ModelCheckpoint(checkpoint_file_format, monitor='val_loss', save_best_only=False,
                                       save_weights_only=True)
    logger_callback = LoggerCallback(logging.getLogger(__name__))

    model.fit_generator(train_data_generator(), steps_per_epoch=train_data_generator.steps_per_epoch,
                        validation_data=validation_data_generator(),
                        validation_steps=validation_data_generator.steps_per_epoch, epochs=epochs,
                        callbacks=[model_checkpoint, logger_callback], verbose=verbose)

    model.save_weights(output_file)


def train_render_network(model, rgb_train_file, brdf_train_file, rgb_validation_file, brdf_validation_file,
                         checkpoint_file_format, output_file, epochs=30, verbose=1):
    """
    Train the RGB->BRDF network.

    :param model: loaded and compiled model
    :param rgb_train_file: name of the RGB training data file
    :param brdf_train_file: name of the BRDF training data file
    :param rgb_validation_file: name of the RGB validation data file
    :param brdf_validation_file: name of the BRDF validation data file
    :param checkpoint_file_format: format of the filename for storing model
    checkpoints, using the str.format() syntax and keras metrics for variables
    :param output_file: output file for the completed model
    :param epochs: number of epochs to train
    :param verbose: same as :func:`keras.models.Model.fit`
    """
    _train_network(model, rgb_train_file, brdf_train_file, rgb_validation_file, brdf_validation_file, 'RGB',
                   'BRDF', checkpoint_file_format, output_file, common.render_data_normalizer, epochs, verbose)


def train_depth_network(model, brdf_train_file, depth_train_file, brdf_validation_file, depth_validation_file,
                        checkpoint_file_format, output_file, epochs=30, verbose=1):
    """
    Train the BRDF->depth network.

    :param model: loaded and compiled model
    :param rgb_train_file: name of the BRDF training data file
    :param brdf_train_file: name of the depth training data file
    :param rgb_validation_file: name of the BRDF validation data file
    :param brdf_validation_file: name of the depth validation data file
    :param checkpoint_file_format: format of the filename for storing model
    checkpoints, using the str.format() syntax and keras metrics for variables
    :param output_file: output file for the completed model
    :param epochs: number of epochs to train
    :param verbose: same as :func:`keras.models.Model.fit`
    """
    _train_network(model, brdf_train_file, depth_train_file, brdf_validation_file, depth_validation_file, 'BRDF',
                   'Z', checkpoint_file_format, output_file, common.depth_data_normalizer, epochs, verbose)
