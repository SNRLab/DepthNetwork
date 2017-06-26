import os.path

import h5py
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

import depth_network.common as common
from depth_network.data_utils import HDFGenerator
from depth_network.model import DepthNetwork
import matplotlib.pyplot as plt


def main():
    rgb_data = h5py.File(common.rgb_data_file, 'r')
    depth_data = h5py.File(common.depth_data_file, 'r')
    data_generator = HDFGenerator(rgb_data['RGB'], depth_data['Z'], batch_size=16, normalizer=common.data_normalizer)

    if os.path.exists(common.model_file):
        model = common.load_model()
    else:
        model = DepthNetwork(input_shape=(3, 64, 64))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=[common.dice_coef, 'accuracy'])

    model_checkpoint = ModelCheckpoint(common.model_file, monitor='loss', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=common.log_dir, histogram_freq=0, batch_size=16, write_graph=True,
                                       write_grads=False, write_images=True, embeddings_freq=0,
                                       embeddings_layer_names=None, embeddings_metadata=None)

    plt.show(block=False)
    model.fit_generator(data_generator(), steps_per_epoch=data_generator.steps_per_epoch, epochs=30,
                        callbacks=[model_checkpoint, tensorboard_callback])


if __name__ == "__main__":
    main()
