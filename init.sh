#!/bin/sh

# init file from previous pilot run
octave --path oct -q --eval "prepare_init;"

# sample parameters for testing
libbi sample --target prior --model-file NPZD.bi --nsamples 16 --output-file data/init_test_filter.nc

# fit bridge weight function
octave --path oct -q --eval "prepare_input;"
