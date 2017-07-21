import argparse
import logging

import h5py
import yaml
from keras.callbacks import ModelCheckpoint, Callback

import depth_network.common as common
from depth_network.data_utils import HDFGenerator

_logger = logging.getLogger(__name__)


class LoggerCallback(Callback):
    def __init__(self, base_logger):
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
                   checkpoint_file_format, output_file, data_normalizer, epochs):
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
    logger_callback = LoggerCallback(_logger)

    model.fit_generator(train_data_generator(), steps_per_epoch=train_data_generator.steps_per_epoch,
                        validation_data=validation_data_generator(),
                        validation_steps=validation_data_generator.steps_per_epoch, epochs=epochs,
                        callbacks=[model_checkpoint, logger_callback])

    model.save_weights(output_file)


def train_render_network(rgb_train_file, brdf_train_file, rgb_validation_file, brdf_validation_file,
                         checkpoint_file_format, output_file, epochs=30):
    render_model = common.load_render_model(create=True)

    _train_network(render_model, rgb_train_file, brdf_train_file, rgb_validation_file, brdf_validation_file, 'RGB',
                   'BRDF', checkpoint_file_format, output_file, common.render_data_normalizer, epochs)


def train_depth_network(brdf_train_file, depth_train_file, brdf_validation_file, depth_validation_file,
                        checkpoint_file_format, output_file, epochs=30):
    depth_model = common.load_depth_model(create=True)

    _train_network(depth_model, brdf_train_file, depth_train_file, brdf_validation_file, depth_validation_file, 'BRDF',
                   'Z', checkpoint_file_format, output_file, common.depth_data_normalizer, epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))
    args = parser.parse_args()

    config = yaml.load(args.config)

    log_file = config.get('log_file', None)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s:%(name)s: %(message)s'))
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    data_config = config.get('data', {})
    train_config = config.get('train', {})
    train_render_config = train_config.get('render', None)
    train_depth_config = train_config.get('depth', None)
    epochs = train_config.get('epochs', 30)

    if train_render_config is not None:
        train_render_network(data_config['rgb']['train_data'], data_config['brdf']['train_data'],
                             data_config['rgb']['validation_data'], data_config['brdf']['validation_data'],
                             train_render_config['checkpoint_file_format'], train_render_config['output_file'], epochs)
    if train_depth_config is not None:
        train_depth_network(data_config['brdf']['train_data'], data_config['depth']['train_data'],
                            data_config['brdf']['validation_data'], data_config['depth']['validation_data'],
                            train_depth_config['checkpoint_file_format'], train_depth_config['output_file'], epochs)


if __name__ == '__main__':
    main()
