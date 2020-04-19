#!/bin/bash
python calibrate.py -i "data/{}_mask_EVERY_MASK.txt"
python calibrate.py -i "data/{}_RGBA+GEOM.txt"
python calibrate.py -i "data/{}_JOIN.txt"
python calibrate.py -i "data/{}_PROB.txt"

python nn_prob_output.py -l GEOM -a GEOM_ -d GEOM_
python calibrate.py -i "data/{}_GEOM.txt"
