#!/bin/bash
title=True
description=True

export PYTHONPATH="/mnt/c/Users/nbaum/Dropbox/wdc-pave/:$PYTHONPATH"

python3 ../analysis/dataset_statistics.py --dataset wdc --title $title --description $description 
