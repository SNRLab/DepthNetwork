import numpy as np


class HDFGenerator:
    def __init__(self, x_dataset, y_dataset, batch_size=16, shuffle=True, normalizer=None):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.start_index = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalizer = normalizer

        assert len(x_dataset) == len(y_dataset)

    def __call__(self):
        while True:
            data_len = len(self.x_dataset)
            end_index = self.start_index + self.batch_size
            wrap_len = 0
            if end_index > data_len:
                wrap_len = end_index - data_len
                end_index = data_len

            x_batch = self.x_dataset[self.start_index: end_index]
            y_batch = self.y_dataset[self.start_index: end_index]

            if end_index == data_len:
                self.start_index = wrap_len
                if wrap_len > 0:
                    x_batch = np.append(x_batch, self.x_dataset[:wrap_len], axis=0)
                    y_batch = np.append(y_batch, self.y_dataset[:wrap_len], axis=0)
            else:
                self.start_index += self.batch_size

            if self.normalizer is not None:
                x_batch, y_batch = self.normalizer(x_batch, y_batch)

            if self.shuffle:
                rng_state = np.random.get_state()
                np.random.shuffle(x_batch)
                np.random.set_state(rng_state)
                np.random.shuffle(y_batch)

            assert len(x_batch) == len(y_batch) == self.batch_size

            # cv2.imshow("RGB", np.transpose(x_batch[10], (1, 2, 0)))
            # cv2.imshow("Depth", cv2.applyColorMap(y_batch[10][0].astype(np.uint8), cv2.COLORMAP_AUTUMN))
            # cv2.waitKey(20)
            yield x_batch, y_batch

    @property
    def steps_per_epoch(self):
        return round(len(self.x_dataset) / self.batch_size)