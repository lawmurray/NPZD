#!/bin/sh

P=10 # no. trajectories
K=365 # no. output points (should match no. records in forcings file at this stage)
T=365.0 # time to simulate (again, should match no. records in forcings file)
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INPUT_FILE=/mnt/data/data/CF1.nc # forcings file
OUTPUT_FILE=/mnt/data/data/output.nc # output file

build/simulate -P $P -K $K -T $T --seed $SEED --input-file $INPUT_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
