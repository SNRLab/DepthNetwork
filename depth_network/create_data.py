import logging
import os.path

import depth_network.common as common
import depth_network.data_utils as data_utils

_logger = logging.getLogger(__name__)


def main():
    rgb_file_names = list(map(lambda n: os.path.join(common.data_dir, n),
                              ['0_rgb_small.hdf5', '2_rgb_small.hdf5', '3_rgb_small.hdf5', '4_rgb_small.hdf5',
                               '5_rgb_small.hdf5', '6_rgb_small.hdf5', '7_rgb_small.hdf5', '8_rgb_small.hdf5',
                               '9_rgb_small.hdf5', '10_rgb_small.hdf5', '11_rgb_small.hdf5', '12_rgb_small.hdf5',
                               '13_rgb_small.hdf5', '14_rgb_small.hdf5', '15_rgb_small.hdf5', '16_rgb_small.hdf5']))
    brdf_file_names = list(map(lambda n: os.path.join(common.data_dir, n),
                               ['0_brdf_small.hdf5', '2_brdf_small.hdf5', '3_brdf_small.hdf5', '4_brdf_small.hdf5',
                                '5_brdf_small.hdf5', '6_brdf_small.hdf5', '7_brdf_small.hdf5', '8_brdf_small.hdf5',
                                '9_brdf_small.hdf5', '10_brdf_small.hdf5', '11_brdf_small.hdf5', '12_brdf_small.hdf5',
                                '13_brdf_small.hdf5', '14_brdf_small.hdf5', '15_brdf_small.hdf5',
                                '16_brdf_small.hdf5']))
    depth_file_names = list(map(lambda n: os.path.join(common.data_dir, n),
                                ['0_z_small.hdf5', '2_z_small.hdf5', '3_z_small.hdf5', '4_z_small.hdf5',
                                 '5_z_small.hdf5', '6_z_small.hdf5', '7_z_small.hdf5', '8_z_small.hdf5',
                                 '9_z_small.hdf5', '10_z_small.hdf5', '11_z_small.hdf5', '12_z_small.hdf5',
                                 '13_z_small.hdf5', '14_z_small.hdf5', '15_z_small.hdf5', '16_z_small.hdf5']))
    data_utils.merge_data_files(common.rgb_data_file, rgb_file_names, 'RGB')
    data_utils.fold_data(common.rgb_data_file, common.rgb_train_data_file, common.rgb_validation_data_file, 'RGB')

    data_utils.merge_data_files(os.path.join(common.data_dir, 'brdf.hdf5'), brdf_file_names, 'BRDF')
    data_utils.fold_data(common.brdf_data_file, common.brdf_train_data_file, common.brdf_validation_data_file, 'BRDF')

    data_utils.merge_data_files(os.path.join(common.data_dir, 'depth.hdf5'), depth_file_names, 'Z')
    data_utils.fold_data(common.depth_data_file, common.depth_train_data_file, common.depth_validation_data_file, 'Z')


if __name__ == '__main__':
    main()
