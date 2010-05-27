#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

#source /home/mur387/init.sh

ROOT=.
LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

P=16 # no. trajectories
K=566 # number of output points
T=565.0 # time for to simulate
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INIT_FILE=$ROOT/data/C7_initHP2.nc # initial values file
FORCE_FILE=$ROOT/data/C7_force_pad.nc # forcings file
OUTPUT_FILE=$ROOT/results/simulate.nc # output file
NS=0 # record number in input files

$ROOT/build/simulate --ns $NS -P $P -K $K -T $T --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
