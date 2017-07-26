#!/usr/bin/env python3

import argparse
import logging
import sys

import depth_network.data_utils as data_utils

_logger = logging.getLogger("fold_hdf5")


def _u(string):
    return bytes(string, 'UTF-8')


def main():
    parser = argparse.ArgumentParser(description="Fold an HDF5 dataset into training and validation sets.")
    parser.add_argument('-t', '--training-file', help="output training file", required=True)
    parser.add_argument('-v', '--validation-file', help="output validation file", required=True)
    parser.add_argument('-d', '--dataset', help="dataset name (e.g. \'RGB\', \'BRDF\', \'Z\')", required=True)
    parser.add_argument('-f', '--folds', help="number of folds", default=5, type=int)
    parser.add_argument('-i', '--validation-fold', help="fold to use as validation set", default=0, type=int)
    parser.add_argument('input_file', help='full dataset')
    args = parser.parse_args()

    if args.validation_fold >= args.folds:
        print("Validation fold must be less than number of folds", file=sys.stderr)
        sys.exit(2)

    data_utils.fold_data(args.input_file, args.training_file, args.validation_file, args.dataset,
                         validation_fold=args.validation_fold, folds=args.folds)


if __name__ == '__main__':
    main()
