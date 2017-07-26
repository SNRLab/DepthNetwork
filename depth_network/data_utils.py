import logging
import math
import random

import numpy as np
from h5py import h5d, h5f, h5s, h5p, h5t
import OpenEXR
import Imath

_logger = logging.getLogger(__name__)


def _u(string):
    return bytes(string, 'UTF-8')


class HDFGenerator:
    """
    Generator that loads data from an HDF5 file and normalizes it.
    """

    def __init__(self, x_dataset, y_dataset, batch_size=16, shuffle=False, normalizer=None):
        """
        Create an HDFGenerator.

        Due to the low performance of the HDF5 library with random access, shuffling causes each batch to consist of a
        contiguous block of samples (taken from a random index place within the dataset), which are then shuffled,
        rather than individual samples chosen randomly from the dataset.

        After :func:`HDFGenerator.steps_per_epoch` iterations, all the data in the dataset is guaranteed to be returned.

        :param x_dataset: name of the x (input) dataset
        :param y_dataset: name of the y (output) dataset
        :param batch_size: number of samples to yield at once
        :param shuffle: if true, shuffle the data
        :param normalizer: normalizer function, which processes entire batches (both x and y) at once
        """
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.cur_index = (len(x_dataset) // batch_size) - 3
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalizer = normalizer

        assert len(x_dataset) == len(y_dataset)

        # cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("BRDF", cv2.WINDOW_NORMAL)

        if shuffle:
            self.shuffle_indices = list(range(int(math.ceil(len(x_dataset) / batch_size))))
            random.shuffle(self.shuffle_indices)

    def __call__(self):
        while True:
            data_len = len(self.x_dataset)
            if self.shuffle:
                start_index = self.shuffle_indices[self.cur_index] * self.batch_size
            else:
                start_index = self.cur_index * self.batch_size
            end_index = start_index + self.batch_size
            if end_index > data_len:
                wrap_len = end_index - data_len
                end_index = data_len
            else:
                wrap_len = 0

            x_batch = self.x_dataset[start_index: end_index]
            y_batch = self.y_dataset[start_index: end_index]

            if end_index == data_len:
                if wrap_len > 0:
                    x_batch = np.append(x_batch, self.x_dataset[:wrap_len], axis=0)
                    y_batch = np.append(y_batch, self.y_dataset[:wrap_len], axis=0)

            self.cur_index += 1
            if self.cur_index * self.batch_size >= data_len:
                self.cur_index = 0
                if self.shuffle:
                    random.shuffle(self.shuffle_indices)

            if self.shuffle:
                p = np.random.permutation(self.batch_size)
                x_batch = x_batch[p]
                y_batch = y_batch[p]

            if self.normalizer is not None:
                x_batch, y_batch = self.normalizer(x_batch, y_batch)

            assert len(x_batch) == len(y_batch) == self.batch_size

            # for image, brdf in zip(x_batch, y_batch):
            #     cv2.imshow("RGB", np.transpose(image, (1, 2, 0)))
            #     cv2.imshow("BRDF", np.transpose(brdf, (1, 2, 0)))
            #     cv2.waitKey()
            yield x_batch, y_batch

    @property
    def steps_per_epoch(self):
        return int(math.ceil(len(self.x_dataset) / self.batch_size))


def merge_data_files(file_path, source_files, dataset_name):
    """
    Merge multiple HDF5 files into one virtual dataset. This virtual dataset appears as one large dataset when used with
    the normal HDF5 APIs, but internally it is referencing each merged file. Virtual datasets are only supported in
    HDF5>=1.10.

    :param file_path: output file path
    :param source_files: list of source files
    :param dataset_name: name of the dataset to merge
    """
    _logger.info("Creating virtual dataset file: %s", file_path)

    files = list(map(lambda f: h5f.open(_u(f), flags=h5f.ACC_RDONLY), source_files))
    datasets = []
    dataspaces = []
    num_elems = 0

    # Virtual datasets are not yet supported in the high-level h5py API, so we have to use the low-level API

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
    """
    Folds a dataset into training and validation sets. The training and validation datasets are virtual, so they do not
    contain a copy of the data. This function can be used for performing k-fold cross validation.

    :param file_path: path to full dataset
    :param train_file_path: output training dataset file
    :param validation_file_path: output validation dataset file
    :param dataset_name: name of the dataset to fold
    :param validation_fold: where to fold the data, 0<=validation_fold<folds
    :param folds: number of folds
    """
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


def read_exr_depth(file):
    file = OpenEXR.InputFile(file)
    data_window = file.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)

    HALF = Imath.PixelType(Imath.PixelType.HALF)
    data = np.expand_dims(np.frombuffer(file.channel('R', HALF), np.float16).reshape(size), -1)

    return data
