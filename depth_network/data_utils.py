import math
import random

import cv2
import numpy as np


class HDFGenerator:
    def __init__(self, x_dataset, y_dataset, batch_size=16, shuffle=False, normalizer=None):
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
