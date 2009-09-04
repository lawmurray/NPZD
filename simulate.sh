#!/bin/bash
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu

ROOT=/home/mur387
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

NS=0
P=10 # no. trajectories
K=365 # no. output points (should match no. records in forcings file at this stage)
T=365.0 # time to simulate (again, should match no. records in forcings file)
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INPUT_FILE=$ROOT/data/test_input.nc # forcings file
OUTPUT_FILE=$ROOT/npzd/results/output.nc # output file

$ROOT/npzd/build/simulate --ns $NS -P $P -K $K -T $T --seed $SEED --input-file $INPUT_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
