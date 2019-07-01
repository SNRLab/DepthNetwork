import os, sys, subprocess
# Loads the weights created by train.sh and allows viewing of the results

os.chdir("../../")
sys.path.append(os.getcwd())
# Add to environment variables beforehand

command = "python tools/test.py --rgb data/paper/rgb.hdf5 --brdf data/paper/brdf.hdf5 --depth data/paper/depth.hdf5 -d data/paper/model/depth.hdf5 -r data/paper/model/render.hdf5 --swap-depth-axes"
os.system(command)