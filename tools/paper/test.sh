#!/bin/bash

# Loads the weights created by train.sh and allows viewing of the results

BASE="$(dirname "$(readlink -f "$0")")"/../..
export PYTHONPATH="${BASE}"

"${BASE}/tools/test.py" --rgb "${BASE}/data/paper/rgb.hdf5" \
                        --brdf "${BASE}/data/paper/brdf.hdf5" \
                        --depth "${BASE}/data/paper/depth.hdf5" \
                        -d "${BASE}/data/paper/model/depth.hdf5" \
                        -r "${BASE}/data/paper/model/render.hdf5" \
                        --swap-depth-axes