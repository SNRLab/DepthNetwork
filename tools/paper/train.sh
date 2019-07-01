#!/bin/bash

# Trains both networks using the sample data from the paper. The data must
# already have been downloaded and prepared.

BASE="$(dirname "$(readlink -f "$0")")"/../..
export PYTHONPATH="${BASE}"

cd "${BASE}"

"${BASE}/tools/train.py" "${BASE}/tools/paper/train.yaml"