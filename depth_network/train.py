import os.path

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import depth_network.common as common
from depth_network.better_io_utils import BetterHDF5Matrix
from depth_network.model import DepthNetwork

smooth = 1.


def main():
    rgb_data = BetterHDF5Matrix(common.rgb_data_file, 'RGB', normalizer=common.preprocess_rgb,
                                normalized_shape=(3,) + common.image_size)
    depth_data = BetterHDF5Matrix(common.depth_data_file, 'Z', normalizer=common.preprocess_depth,
                                  normalized_shape=(16,) + common.image_size)

    if os.path.exists(common.model_file):
        model = common.load_model()
    else:
        model = DepthNetwork(input_shape=(3, 64, 64))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=[common.dice_coef, 'accuracy'])

    model_checkpoint = ModelCheckpoint(common.model_file, monitor='val_loss', save_best_only=True)

    model.fit(rgb_data, depth_data, batch_size=16, epochs=10, shuffle='batch', validation_split=0.3,
              callbacks=[model_checkpoint])

if __name__ == "__main__":
    main()
