#!/bin/bash

cp lib/model_optimized.py model_optimized.pyx
cp lib/hmm_optimized.py hmm_optimized.pyx
python setup.py build_ext --inplace
