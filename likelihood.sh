#!/bin/bash
#PBS -l walltime=01:00:00,nodes=1:ppn=8,vmem=8gb
#PBS -t 0-49
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    source init.sh
    ROOT=$PBS_O_WORKDIR
    TMP_DIR=$MEMDIR
    ID=${PBS_JOBNAME}_$PBS_ARRAYID
else
    ROOT=.
    TMP_DIR=/tmp
    ID=mcmc
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=2 # no. OpenMP threads per process
NPERNODE=1 # no. processes per node

##
## Run config
##

# MCMC settings
T=365.0 # time to simulate
SEED=20 # pseudorandom number seed
C=100 # no. samples to draw

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=1024 # no. trajectories
L=1 # lookahead for auxiliary particle filter

# input files, in $DATA_DIR
INIT_FILE=PZtest1_init.nc # initial values file
FORCE_FILE=C7_force.nc # forcings file
OBS_FILE=PZtest1_obs_gaussian.nc # observations file

# output file, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc

# intermediate results file 
FILTER_FILE=$TMP_DIR/pf.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=1

# copy forcings and observations to local directory, used repeatedly
mpirun -npernode 1 cp $DATA_DIR/$FORCE_FILE $DATA_DIR/$OBS_FILE $TMP_DIR/.

# output this script as record of settings
cat $ROOT/likelihood.sh

mpirun -npernode $NPERNODE $ROOT/build/likelihood -P $P -T $T --resampler $RESAMPLER -L $L -C $C --seed $SEED --init-file $DATA_DIR/$INIT_FILE --force-file $TMP_DIR/$FORCE_FILE --obs-file $TMP_DIR/$OBS_FILE --filter-file $FILTER_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $TMP_DIR/$OUTPUT_FILE

# copy results from $TMPDIR to $RESULTS_DIR
mpirun -npernode 1 sh -c "cp $TMP_DIR/$OUTPUT_FILE"'*'" $RESULTS_DIR/."
chmod 644 $RESULTS_DIR/*
