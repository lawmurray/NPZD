#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu

module load cuda boost boost-bindings netcdf gsl atlas intel-cc

ROOT=/home/mur387
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

NS=0
P=10 # no. trajectories
K=100 # size of intermediate result buffer
T=365.0 # time to simulate
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INIT_FILE=$ROOT/npzd/data/input_OSP_0D.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/input_OSP_0D.nc # forcings file
OBS_FILE=$ROOT/npzd/data/obs_NEQ.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/output.nc # output file

$ROOT/npzd/build/simulate --ns $NS -P $P -K $K -T $T --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
