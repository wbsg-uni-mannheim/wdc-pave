#!/bin/bash

export PYTHONPATH="/mnt/c/Users/nbaum/Dropbox/wdc-pave/:$PYTHONPATH"

# (a) Split to test/train
python3 ../preprocessing/prepare_datasets.py --dataset wdc

# (b) describe dataset
#python3 ../preprocessing/describe_dataset.py --dataset wdc 

# (c) detect data types
#python3 ../analysis/data_types.py --dataset wdc

