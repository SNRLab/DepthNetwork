#!/bin/bash

BASE="$(dirname "$(readlink -f "$0")")"/../..
export PYTHONPATH="${BASE}"

cd "${BASE}"

"${BASE}/tools/train.py" "${BASE}/tools/paper/train.yaml"