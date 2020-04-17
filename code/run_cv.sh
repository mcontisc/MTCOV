#!/bin/bash

for c in 2 5 8; do {
echo $c
    for g in 0.3 0.5 0.7; do
	python3 main_cv_single_real.py -C=$c -g=$g 
    done;} &
done;