#!/bin/bash

# Merges sample data files and folds datasets into training and validation sets. The data must already be downloaded
# using download_data.sh

BASE="$(dirname "$(readlink -f "$0")")"/../..
export PYTHONPATH="${BASE}"

# Create directories needed for training and cross validation
mkdir -p "${BASE}/data/paper/model/cross_validation"


"${BASE}/tools/merge_hdf5.py" -d RGB -o "${BASE}/data/paper/rgb.hdf5" "${BASE}"/data/paper/*_rgb_small.hdf5
"${BASE}/tools/fold_hdf5.py" -d RGB -t "${BASE}/data/paper/rgb_train.hdf5" -v "${BASE}/data/paper/rgb_validation.hdf5" "${BASE}/data/paper/rgb.hdf5"

"${BASE}/tools/merge_hdf5.py" -d BRDF -o "${BASE}/data/paper/brdf.hdf5" "${BASE}"/data/paper/*_brdf_small.hdf5
"${BASE}/tools/fold_hdf5.py" -d BRDF -t "${BASE}/data/paper/brdf_train.hdf5" -v "${BASE}/data/paper/brdf_validation.hdf5" "${BASE}/data/paper/brdf.hdf5"

"${BASE}/tools/merge_hdf5.py" -d Z -o "${BASE}/data/paper/depth.hdf5" "${BASE}"/data/paper/*_z_small.hdf5
"${BASE}/tools/fold_hdf5.py" -d Z -t "${BASE}/data/paper/depth_train.hdf5" -v "${BASE}/data/paper/depth_validation.hdf5" "${BASE}/data/paper/depth.hdf5"
