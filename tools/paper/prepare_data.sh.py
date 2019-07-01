import os, sys, subprocess, glob

# Merges sample data files and folds datasets into training and validation sets. The data must already be downloaded
# using download_data.sh

os.chdir("../../")
sys.path.append(os.getcwd())
# Add to environment variables beforehand

if not os.path.exists("data/paper/model"):
  os.mkdir("data/paper/model/")
if not os.path.exists("data/paper/model/cross_validation"):
  os.mkdir("data/paper/model/cross_validation")

command1 = "python tools/merge_hdf5.py -d RGB -o data/paper/rgb.hdf5 " + ' '.join(glob.glob('data/paper/*_rgb_small.hdf5')).replace('\\', '/')
os.system(command1)
command2 = "python tools/fold_hdf5.py -d RGB -t data/paper/rgb_train.hdf5 -v data/paper/rgb_validation.hdf5 data/paper/rgb.hdf5"
os.system(command2)

command3 = "python tools/merge_hdf5.py -d BRDF -o data/paper/brdf.hdf5 " + ' '.join(glob.glob('data/paper/*_brdf_small.hdf5')).replace('\\', '/')
os.system(command3)
command4 = "python tools/fold_hdf5.py -d BRDF -t data/paper/brdf_train.hdf5 -v data/paper/brdf_validation.hdf5 data/paper/brdf.hdf5"
os.system(command4)

command5 = "python tools/merge_hdf5.py -d Z -o data/paper/depth.hdf5 " + ' '.join(glob.glob('data/paper/*_z_small.hdf5')).replace('\\', '/')
os.system(command5)
command6 = "python tools/fold_hdf5.py -d Z -t data/paper/depth_train.hdf5 -v data/paper/depth_validation.hdf5 data/paper/depth.hdf5"
os.system(command6)