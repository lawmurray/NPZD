#!/bin/sh
#PBS -l walltime=10:00,nodes=1:ppn=1,gres=gpu
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    source init.sh
    ROOT=$PBS_O_WORKDIR
    NAME=$PBS_JOBNAME
else
    ROOT=.
    NAME=pf
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=4

##
## Run config
##

P=1024 # no. trajectories
T=365.0 # time to simulate
H=1.0 # initial step size for ODE integrator
ATOLER=1.0e-3 # absolute error tolerance for ODE integrator
RTOLER=1.0e-3 # relative error tolerance for ODE integrator
SEED=3 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=1 # produce timings?
RESAMPLER=stratified
L=0

INIT_FILE=$DATA_DIR/C7_initHP2.nc # initial values file
FORCE_FILE=$DATA_DIR/C7_force.nc # forcings file
OBS_FILE=$DATA_DIR/C7_S1_obs_HP2.nc # observations file
OUTPUT_FILE=$RESULTS_DIR/${NAME}.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/build/pf -P $P -T $T -h $H --atoler=$ATOLER --rtoler=$RTOLER --seed=$SEED --resampler=$RESAMPLER -L $L --init-file=$INIT_FILE --force-file=$FORCE_FILE --obs-file=$OBS_FILE --init-ns=$INIT_NS --force-ns=$FORCE_NS --obs-ns=$OBS_NS --output-file=$OUTPUT_FILE --output=$OUTPUT --time=$TIME
