#!/bin/sh

# sample parameters for testing
libbi sample --target prior --model-file NPZD.bi --nsamples 50 --output-file data/init_osp.nc

# fit bridge weight function
octave --path oct -q --eval "prepare_input;"
