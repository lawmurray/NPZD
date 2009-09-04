#!/bin/bash
#PBS –l walltime=05:00,nodes=1:ppn=1,gres=gpu -d $PBS_O_WORKDIR

LD_LIBRARY_PATH=../bi/build:$LD_LIBRARY_PATH

P=10 # no. trajectories
K=365 # no. output points (should match no. records in forcings file at this stage)
T=365.0 # time to simulate (again, should match no. records in forcings file)
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INPUT_FILE=data/test_input_LC.nc # forcings file
OUTPUT_FILE=results/output.nc # output file

build/simulate -P $P -K $K -T $T --seed $SEED --input-file $INPUT_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
