#!/bin/sh
#PBS -l walltime=24:00:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe
#PBS -t 0-4

#module load cuda boost boost-bindings netcdf gsl atlas intel-cc

ROOT=/home/mur387/workspace
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

P=1024 # no. trajectories
T=365.0 # time to simulate
H=0.3 # initial step size
SCALE=0.05 # scale of proposal relative to prior
B=0 # burn in steps
I=1 # sample interval (in steps)
L=10 # no. samples to draw
TEMP=2.0
MIN_ESS=1.0
SEED=2 # pseudorandom number seed

INIT_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # forcings file
OBS_FILE=$ROOT/npzd/data/GPUobs_EQ.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/mcmc.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/npzd/build/mcmc -P $P -T $T -h $H --temp $TEMP --scale $SCALE --min-ess $MIN_ESS -B $B -I $I -L $L --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE

