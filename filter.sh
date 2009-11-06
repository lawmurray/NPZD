#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

module load cuda boost boost-bindings netcdf gsl atlas intel-cc

ROOT=/home/mur387
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

P=4096 # no. trajectories
K=8 # size of intermediate result buffer
T=365.0 # time to simulate
H=0.3 # initial step size
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INIT_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # forcings file
OBS_FILE=$ROOT/npzd/data/GPUobs_NEQ.nc # observations file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/npzd/build/filter -P $P -K $K -T $T -h $H --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME
