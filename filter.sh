#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

#module load cuda boost boost-bindings netcdf gsl atlas intel-cc

ROOT=/home/mur387/workspace
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

P=1024 # no. trajectories

T=565.0 # time to simulate
H=0.3 # initial step size
SEED=3 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=1 # produce timings?
MIN_ESS=1.0
RESAMPLER=stratified
L=20

INIT_FILE=$ROOT/npzd/data/C7_init.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/C7_force_pad.nc # forcings file
OBS_FILE=$ROOT/npzd/data/C7_obs_pad.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/pf.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/npzd/build/filter -P $P -T $T -h $H --min-ess $MIN_ESS --seed $SEED --resample $RESAMPLER -L $L --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file ${OUTPUT_FILE} --output $OUTPUT --time $TIME
