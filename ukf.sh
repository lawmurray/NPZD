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
    NAME=ukf
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=2

##
## Run config
##

T=365.0 # time to simulate
H=0.3 # initial step size
SEED=3 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=1 # produce timings?

# input files, in $DATA_DIR
INIT_FILE=$DATA_DIR/C7_init.nc # initial values file
FORCE_FILE=$DATA_DIR/C7_force_pad.nc # forcings file
OBS_FILE=$DATA_DIR/C7_S1_obs_HP2.nc # observations file
OUTPUT_FILE=$RESULTS_DIR/$NAME.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=1

$ROOT/build/ukf -T $T -h $H --seed=$SEED --init-file=$INIT_FILE --force-file=$FORCE_FILE --obs-file=$OBS_FILE --init-ns=$INIT_NS --force-ns=$FORCE_NS --obs-ns=$OBS_NS --output-file=$OUTPUT_FILE --output=$OUTPUT --time=$TIME
