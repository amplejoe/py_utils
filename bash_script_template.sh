#!/bin/bash

prog=${0##*/}

# augments full dataset DISREGARDING splits
if [ $# -ne 2 ]; then
    echo "usage: bash augment_full_ds.sh [IN_PATH] [OUT_PATH] [NUM_AUGMENTS_PER_IMAGE]"
    echo "encapsule paths within '' if they contain spaces"
    echo "output: [IN_PATH]_{various_augmentation_options}"
    echo "[AUG_OPTIONS] are comma separated. See dataset_processing/augmentation/process_ds.py for all options."
    echo ""
    echo "Script help:"
    python ../../dataset_processing/augmentation/process_ds.py -h
    exit 1
fi

# reads path without '/'
IN_PATH=${1%/}
OUT_PATH=${2%/}