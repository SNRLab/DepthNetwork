import h5py
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard

import depth_network.common as common
from depth_network.data_utils import HDFGenerator


def main():
    rgb_data = h5py.File(common.rgb_data_file, 'r')
    depth_data = h5py.File(common.depth_data_file, 'r')
    data_generator = HDFGenerator(rgb_data['RGB'], depth_data['Z'], batch_size=16, normalizer=common.data_normalizer)

    render_model, depth_model = common.load_models(create=True)

    render_model_checkpoint = ModelCheckpoint(common.render_model_file, monitor='loss', save_best_only=True)
    depth_model_checkpoint = ModelCheckpoint(common.depth_model_file, monitor='loss', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=common.log_dir, histogram_freq=0, batch_size=16, write_graph=True,
                                       write_grads=False, write_images=True, embeddings_freq=0,
                                       embeddings_layer_names=None, embeddings_metadata=None)

    plt.show(block=False)
    depth_model.fit_generator(data_generator(), steps_per_epoch=data_generator.steps_per_epoch, epochs=30,
                        callbacks=[depth_model_checkpoint])


if __name__ == "__main__":
    main()
