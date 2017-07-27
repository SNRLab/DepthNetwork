# DepthNetwork

An implementation of [Deep Monocular 3D Reconstruction for Assisted
Navigation in Bronchoscopy](http://www.marcovs.com/bronchoscopy-navigation/)
in Python using [Keras](https://keras.io/).

Keras supports multiple backends, and this implementation has only been
tested with TensorFlow, but it might work with others, though it may
need minor modifications.

The actual implementation of the network and other related library
functions can be found in the `depth_network/` folder. All executable
scripts are located in the `tools/` folder.

More information about each component can be found in comments at the
top of the files.

### Dependencies
* [Python 3](https://www.python.org/)
* [Keras](https://keras.io/)
* [scikit-image](http://scikit-image.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [h5py](http://www.h5py.org/)
* [matplotlib](https://matplotlib.org/)
* [PyYAML](http://pyyaml.org/wiki/PyYAML)
* [openexrpython](https://github.com/jamesbowman/openexrpython)
  (optional, for converting depth maps from the Unity renderer)

## Using sample data from paper

A set of shell scripts are provided in the `tools/paper` directory that
will automatically download and prepare the data collected by
Visentini-Scarzanella et al., as well as train and test the network.

Run the scripts in this order:

`download_data.sh`

`prepare_data.sh`

`train.sh`

`test.sh`