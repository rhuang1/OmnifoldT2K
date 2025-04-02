#!/bin/bash

module load tensorflow/2.9.0
python t2k.py --config config_omnifold.json --weights_folder weights_omnifold/ --file_path /global/cfs/projectdirs/m4045/users/rhuang/T2K/onoffaxisdata/FormattedData_v13/ --no_eff --verbose
