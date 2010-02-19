#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

#source /home/mur387/init.sh

ROOT=/home/mur387/workspace
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

T=565 # time to simulate
H=0.3 # initial step size
SEED=3 # pseudorandom number seed
OUTPUT=0 # produce output?
TIME=1 # produce timings?

INIT_FILE=$ROOT/npzd/data/C7_init.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/C7_force_pad.nc # forcings file
OBS_FILE=$ROOT/npzd/data/C7_obs_pad.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/ukf.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=0

$ROOT/npzd/build/ukf -T $T -h $H --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file ${OUTPUT_FILE} --output $OUTPUT --time $TIME
