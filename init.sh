#!/bin/sh

# sample parameters for testing
libbi sample --target prior --model-file NPZD.bi --nsamples 1024 --output-file data/init.nc

# fit bridge weight function
octave --path oct -q --eval "prepare_input;"
