#!/bin/bash

module load tensorflow/2.9.0
python t2k.py --config config_omnifold.json --weights_folder weights_omnifold/ --file_path FormattedData/ --no_eff --verbose
