#!/bin/sh
#PBS -l walltime=24:00:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe
#PBS -t 0-4

#module load cuda boost boost-bindings netcdf gsl atlas intel-cc

ROOT=/home/mur387/workspace
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

PBS_ARRAYID=0
P=4096 # no. trajectories
T=10.0 # time to simulate
H=0.3 # initial step size
SCALE=0.05 # scale of proposal relative to prior
B=0 # burn in steps
I=1 # sample interval (in steps)
L=3 # no. samples to draw
TEMPERATURE=1.0
MIN_ESS=1.0

SEED=$PBS_ARRAYID # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=0 # produce timings?

INIT_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # forcings file
OBS_FILE=$ROOT/npzd/data/GPUobs_EQ.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/mcmc$PBS_ARRAYID.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/npzd/build/mcmc -P $P -T $T -h $H --temperature $TEMPERATURE --scale $SCALE --min-ess $MIN_ESS -B $B -I $I -L $L --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME

