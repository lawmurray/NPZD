#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

source /home/mur387/init.sh

ROOT=/home/mur387
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

P=1024 # no. trajectories
K=565 # size of intermediate result buffer
T=565.0 # time to simulate
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INIT_FILE=$ROOT/npzd/data/GPUinput_OSP_C7_init.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/GPUinput_OSP_C7_force.nc # forcings file
OUTPUT_FILE=$ROOT/npzd/results/output.nc # output file
NS=0 # record number in input files

$ROOT/npzd/build/simulate --ns $NS -P $P -K $K -T $T --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
