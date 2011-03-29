#!/bin/bash
#PBS -l walltime=24:00:00,nodes=1:ppn=4,vmem=16gb
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    source init.sh
    ROOT=$PBS_O_WORKDIR
    LOCAL_DIR=$TMPDIR
    NAME=$PBS_JOBNAME
    ID=$PBS_ARRAYID
else
    ROOT=.
    LOCAL_DIR=/tmp
    NAME=likelihood
    ID=0
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=4 # no. OpenMP threads per process
NPERNODE=1 # no. processes per node

##
## Run config
##

# MCMC settings
T=365.0 # time to simulate
SEED=0 # pseudorandom number seed
C=10 # no. samples to draw
M=1 # no. steps before changing sample

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=1024 # no. trajectories

# ODE settings
H=1.0 # initial step size for ODE integrator
ATOLER=1.0e-3 # absolute error tolerance for ODE integrator
RTOLER=1.0e-3 # relative error tolerance for ODE integrator

# input files, in $DATA_DIR
INIT_FILENAME=C7_initHP2.nc
FORCE_FILENAME=C7_force.nc
OBS_FILENAME=C7_S1_obs_HP2.nc
FILTER_FILENAME=pf.nc.$ID
PROPOSAL_FILENAME=urts.nc
OUTPUT_FILENAME=$NAME.nc.$ID

INIT_FILE=$DATA_DIR/$INIT_FILENAME # initial values file
FORCE_FILE=$DATA_DIR/$FORCE_FILENAME # forcings file
OBS_FILE=$DATA_DIR/$OBS_FILENAME # observations file
PROPOSAL_FILE= # proposal file
FILTER_FILE=$LOCAL_DIR/$FILTER_FILENAME # intermediate filter results
PROPOSAL_FILE=$RESULTS_DIR/$PROPOSAL_FILENAME # URTSS results
OUTPUT_FILE=$LOCAL_DIR/$OUTPUT_FILENAME # output file

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=1

# copy forcings and observations to local file system
#pbsdsh -u
cp $INIT_FILE $FORCE_FILE $OBS_FILE $LOCAL_DIR/.

# output this script as record of settings
#cat $ROOT/npzd/mcmc.sh

# run
$ROOT/build/likelihood --id=$ID -P $P -T $T --resampler=$RESAMPLER -C $C -M $M -h $H --atoler=$ATOLER --rtoler=$RTOLER --seed=$SEED --init-file=$LOCAL_DIR/$INIT_FILENAME --force-file=$LOCAL_DIR/$FORCE_FILENAME --obs-file=$LOCAL_DIR/$OBS_FILENAME --filter-file=$FILTER_FILE --proposal-file=$PROPOSAL_FILE --init-ns=$INIT_NS --force-ns=$FORCE_NS --obs-ns=$OBS_NS --output-file=$OUTPUT_FILE

# copy results from local to network file system
cp $LOCAL_DIR/$OUTPUT_FILENAME $RESULTS_DIR/.
