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
    ID=$PBS_JOBNAME
else
    ROOT=.
    ID=ukf
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=2

##
## Run config
##

T=200.0 # time to simulate
H=0.3 # initial step size
SEED=3 # pseudorandom number seed
OUTPUT=0 # produce output?
TIME=1 # produce timings?

# input files, in $DATA_DIR
INIT_FILE=C7_init.nc # initial values file
FORCE_FILE=C7_force_pad.nc # forcings file
OBS_FILE=C7_S1_obs_padHP2.nc # observations file
#FORCE_FILE=OSP_71_76_force_pad.nc # forcings file
#OBS_FILE=OSP_71_76_obs_pad.nc # observations file

# output file, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=1

echo $ROOT/build/ukf -T $T -h $H --seed $SEED --init-file $DATA_DIR/$INIT_FILE --force-file $DATA_DIR/$FORCE_FILE --obs-file $DATA_DIR/$OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $RESULTS_DIR/$OUTPUT_FILE --output $OUTPUT --time $TIME
