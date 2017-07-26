#!/usr/bin/env python3

import argparse

import depth_network.data_utils as data_utils


def main():
    parser = argparse.ArgumentParser(description="Merge multiple HDF5 files into one large virtual dataset.")
    parser.add_argument('-o', '--output-file', help="merged output file", required=True)
    parser.add_argument('-d', '--dataset', help="dataset name (e.g. \'RGB\', \'BRDF\', \'Z\')", required=True)
    parser.add_argument('input_files', help='files to merge', nargs='+')
    args = parser.parse_args()

    data_utils.merge_data_files(args.output_file, args.input_files, args.dataset)


if __name__ == '__main__':
    main()
