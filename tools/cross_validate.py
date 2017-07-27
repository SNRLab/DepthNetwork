#!/usr/bin/env python3

import argparse
import logging
import os
import os.path

import yaml

import depth_network.data_utils as data_utils
import depth_network.training as train

logging.basicConfig(level=logging.INFO)


def main():
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Perform k-fold cross validation of the networks.")
    parser.add_argument('config', type=argparse.FileType('r'))
    args = parser.parse_args()

    config = yaml.load(args.config)

    log_file = config.get('log_file', None)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s:%(name)s: %(message)s'))
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    folds = config['folds']
    output_dir = config['output_dir']
    rgb_data_file = config['rgb_data']
    brdf_data_file = config['brdf_data']
    depth_data_file = config['depth_data']

    for i in range(folds):
        logger.info("Started fold %d", i)
        fold_output_dir = os.path.join(output_dir, '{:02d}'.format(i))
        os.mkdir(fold_output_dir)

        rgb_train_data_file = os.path.join(fold_output_dir, 'rgb_train.hdf5')
        rgb_validation_data_file = os.path.join(fold_output_dir, 'rgb_validation.hdf5')
        data_utils.fold_data(rgb_data_file, rgb_train_data_file, rgb_validation_data_file, 'RGB', i, folds)

        brdf_train_data_file = os.path.join(fold_output_dir, 'brdf_train.hdf5')
        brdf_validation_data_file = os.path.join(fold_output_dir, 'brdf_validation.hdf5')
        data_utils.fold_data(brdf_data_file, brdf_train_data_file, brdf_validation_data_file, 'BRDF', i, folds)

        depth_train_data_file = os.path.join(fold_output_dir, 'depth_train.hdf5')
        depth_validation_data_file = os.path.join(fold_output_dir, 'depth_validation.hdf5')
        data_utils.fold_data(depth_data_file, depth_train_data_file, depth_validation_data_file, 'Z', i, folds)

        train.train_render_network(rgb_train_data_file, brdf_train_data_file, rgb_validation_data_file,
                                   brdf_validation_data_file,
                                   os.path.join(fold_output_dir, 'render_model_{epoch:02d}.hdf5'),
                                   os.path.join(fold_output_dir, 'render.hdf5'))

        train.train_depth_network(brdf_train_data_file, depth_train_data_file, brdf_validation_data_file,
                                  depth_validation_data_file,
                                  os.path.join(fold_output_dir, 'depth_model_{epoch:02d}.hdf5'),
                                  os.path.join(fold_output_dir, 'depth.hdf5'))


if __name__ == '__main__':
    main()
