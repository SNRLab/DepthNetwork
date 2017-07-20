import h5py
from keras.callbacks import ModelCheckpoint, TensorBoard

import depth_network.common as common
from depth_network.data_utils import HDFGenerator


def train_render_network():
    rgb_train_data = h5py.File(common.rgb_train_data_file, 'r')
    brdf_train_data = h5py.File(common.brdf_train_data_file, 'r')
    rgb_validation_data = h5py.File(common.rgb_validation_data_file, 'r')
    brdf_validation_data = h5py.File(common.brdf_validation_data_file, 'r')
    train_data_generator = HDFGenerator(rgb_train_data['RGB'], brdf_train_data['BRDF'], batch_size=16,
                                        normalizer=common.render_data_normalizer, shuffle=True)
    validation_data_generator = HDFGenerator(rgb_validation_data['RGB'], brdf_validation_data['BRDF'], batch_size=64,
                                             normalizer=common.render_data_normalizer, shuffle=True)

    render_model = common.load_render_model(create=True)

    model_checkpoint = ModelCheckpoint(common.render_model_checkpoint_file, monitor='dice_coef', save_best_only=False,
                                       save_weights_only=True)
    tensorboard = TensorBoard(common.log_dir, batch_size=1, write_images=True)

    render_model.fit_generator(train_data_generator(), steps_per_epoch=train_data_generator.steps_per_epoch,
                               validation_data=validation_data_generator,
                               validation_steps=validation_data_generator.steps_per_epoch, epochs=30,
                               callbacks=[model_checkpoint, tensorboard])

    render_model.save_weights(common.render_model_file)


def train_depth_network():
    brdf_train_data = h5py.File(common.brdf_train_data_file, 'r')
    depth_train_data = h5py.File(common.depth_train_data_file, 'r')
    brdf_validation_data = h5py.File(common.brdf_validation_data_file, 'r')
    depth_validation_data = h5py.File(common.depth_validation_data_file, 'r')
    train_data_generator = HDFGenerator(brdf_train_data['BRDF'], depth_train_data['Z'], batch_size=16,
                                        normalizer=common.depth_data_normalizer, shuffle=True)
    validation_data_generator = HDFGenerator(brdf_validation_data['BRDF'], depth_validation_data['Z'], batch_size=64,
                                             normalizer=common.depth_data_normalizer, shuffle=False)

    depth_model = common.load_depth_model(create=True)

    model_checkpoint = ModelCheckpoint(common.depth_model_checkpoint_file, monitor='dice_coef', save_best_only=False,
                                       save_weights_only=True)

    depth_model.fit_generator(train_data_generator(), steps_per_epoch=train_data_generator.steps_per_epoch,
                              validation_data=validation_data_generator(),
                              validation_steps=validation_data_generator.steps_per_epoch, epochs=30,
                              callbacks=[model_checkpoint])

    depth_model.save_weights(common.depth_model_file)
