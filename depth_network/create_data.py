import logging
import os.path

from h5py import h5d, h5f, h5s, h5p, h5t

import depth_network.common as common

_logger = logging.getLogger(__name__)


def _u(string):
    return bytes(string, 'UTF-8')


def merge_data_files(file_path, source_files, dataset_name):
    _logger.info("Creating virtual dataset file: %s", file_path)

    files = list(map(lambda f: h5f.open(_u(f), flags=h5f.ACC_RDONLY), source_files))
    datasets = []
    dataspaces = []
    num_elems = 0

    for f in files:
        dataset = h5d.open(f, _u(dataset_name))
        datasets.append(dataset)
        dataspace = dataset.get_space()
        dataspace.select_all()
        dataspaces.append(dataspace)
        num_elems += dataspace.shape[0]
        dims = dataspace.shape[1:]

    virtual_file = h5f.create(_u(file_path))
    virtual_dataspace = h5s.create_simple((num_elems,) + dims)

    dcpl = h5p.create(h5p.DATASET_CREATE)
    starting_elem = 0
    for file_name, dataset, dataspace in zip(source_files, datasets, dataspaces):
        start = (starting_elem, 0, 0, 0)
        virtual_dataspace.select_hyperslab(start, (1, 1, 1, 1), block=dataspace.shape)
        dcpl.set_virtual(virtual_dataspace, _u(file_name), _u(dataset_name), dataspace)
        starting_elem += dataspace.shape[0]

    h5d.create(virtual_file, _u(dataset_name), h5t.NATIVE_FLOAT, virtual_dataspace, dcpl=dcpl).close()

    virtual_dataspace.close()
    virtual_file.close()

    for file, dataset, dataspace in zip(files, datasets, dataspaces):
        dataspace.close()
        dataset.close()
        file.close()


def fold_data(file_path, train_file_path, validation_file_path, dataset_name, validation_fold=0, folds=5):
    file = h5f.open(_u(file_path))
    dataset = h5d.open(file, _u(dataset_name))
    dataspace = dataset.get_space()
    fold_size = round(dataspace.shape[0] / folds)
    num_validation = min(dataspace.shape[0] - (validation_fold * fold_size), fold_size)
    num_train_before = fold_size * validation_fold
    num_train_after = dataspace.shape[0] - num_validation - num_train_before
    _logger.info("Folding dataset %s: | train: %d | val: %d | train: %d |", file_path, num_train_before, num_validation,
                num_train_after)
    assert num_validation + num_train_after + num_train_before == dataspace.shape[0]
    assert num_validation > 0
    assert (num_train_before >= 0 and num_train_after > 0) or (num_train_before > 0 and num_train_after >= 0)

    # Create train file
    train_file = h5f.create(_u(train_file_path))
    train_dataspace = h5s.create_simple((num_train_before + num_train_after,) + dataspace.shape[1:])
    train_before_dataspace_shape = (num_train_before,) + dataspace.shape[1:]
    train_after_dataspace_shape = (num_train_after,) + dataspace.shape[1:]
    # Select beginning and end of dataset
    dataspace.select_hyperslab((0, 0, 0, 0), (1, 1, 1, 1), block=train_before_dataspace_shape)
    dataspace.select_hyperslab((num_train_before + num_validation, 0, 0, 0), (1, 1, 1, 1),
                               block=train_after_dataspace_shape, op=h5s.SELECT_OR)
    train_dataspace.select_hyperslab((0, 0, 0, 0), (1, 1, 1, 1),
                                     block=(num_train_before + num_train_after,) + dataspace.shape[1:])

    train_dcpl = h5p.create(h5p.DATASET_CREATE)
    train_dcpl.set_virtual(train_dataspace, _u(file_path), _u(dataset_name), dataspace)
    h5d.create(train_file, _u(dataset_name), h5t.NATIVE_FLOAT, train_dataspace, dcpl=train_dcpl).close()

    train_dataspace.close()
    train_file.close()

    # Create validation file
    validation_file = h5f.create(_u(validation_file_path))
    validation_dataspace_shape = (num_validation,) + dataspace.shape[1:]
    validation_dataspace = h5s.create_simple(validation_dataspace_shape)

    dataspace.select_hyperslab((num_train_before, 0, 0, 0), (1, 1, 1, 1), block=validation_dataspace_shape)

    validation_dcpl = h5p.create(h5p.DATASET_CREATE)
    validation_dcpl.set_virtual(validation_dataspace, _u(file_path), _u(dataset_name), dataspace)
    h5d.create(validation_file, _u(dataset_name), h5t.NATIVE_FLOAT, validation_dataspace, dcpl=validation_dcpl).close()

    validation_dataspace.close()
    validation_file.close()

    dataspace.close()
    file.close()


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
    merge_data_files(common.rgb_data_file, rgb_file_names, 'RGB')
    fold_data(common.rgb_data_file, common.rgb_train_data_file, common.rgb_validation_data_file, 'RGB')

    merge_data_files(os.path.join(common.data_dir, 'brdf.hdf5'), brdf_file_names, 'BRDF')
    fold_data(common.brdf_data_file, common.brdf_train_data_file, common.brdf_validation_data_file, 'BRDF')

    merge_data_files(os.path.join(common.data_dir, 'depth.hdf5'), depth_file_names, 'Z')
    fold_data(common.depth_data_file, common.depth_train_data_file, common.depth_validation_data_file, 'Z')


if __name__ == '__main__':
    main()
