#!/usr/bin/env python3

import argparse
import logging

import yaml

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Train one or both networks according to the specified config file.")
    parser.add_argument('config', help="configuration file", type=argparse.FileType('r'))
    parser.add_argument('--continue', help="continue training from the specified epoch", type=int,
                        dest='continue_epoch')
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

    if args.continue_epoch is not None:
        if args.continue_epoch <= 0:
            parser.error("--continue must be greater than 0")
        if args.continue_epoch > epochs:
            parser.error("--continue must be less than or equal to the number of epochs")
        epochs -= (args.continue_epoch - 1)

    import depth_network.common as common
    from depth_network.training import train_depth_network, train_render_network

    if train_render_config is not None:
        render_output_file = train_render_config['output_file']

        if args.continue_epoch is not None:
            render_model = common.load_render_model(render_output_file, create=False)
        else:
            render_model = common.load_render_model(None, create=True)

        train_render_network(render_model, data_config['rgb']['train_data'], data_config['brdf']['train_data'],
                             data_config['rgb']['validation_data'], data_config['brdf']['validation_data'],
                             train_render_config['checkpoint_file_format'], render_output_file, epochs, verbose=1)

    if train_depth_config is not None:
        depth_output_file = train_depth_config['output_file']

        if args.continue_epoch is not None:
            depth_model = common.load_depth_model(depth_output_file, create=False)
        else:
            depth_model = common.load_depth_model(None, create=True)

        train_depth_network(depth_model, data_config['brdf']['train_data'], data_config['depth']['train_data'],
                            data_config['brdf']['validation_data'], data_config['depth']['validation_data'],
                            train_depth_config['checkpoint_file_format'], depth_output_file, epochs, verbose=1)


if __name__ == '__main__':
    main()
