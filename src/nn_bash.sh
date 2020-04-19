#!/bin/bash
python nn_prob_output.py -l RGBA -a RGBA_ -d Q_
python nn_prob_output.py -l RGBA+GEOM -a RGBA_ GEOM_ -d Q_ GEOM_
python nn_prob_output.py -l JOIN -a JOIN_ -d Q_ GEOM_
python nn_prob_output.py -l PROB -a PROB_ -d Q_ GEOM_

