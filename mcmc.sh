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
    MEM_DIR=$MEMDIR
    TMP_DIR=$TMPDIR
    ID=$PBS_JOBNAME
else
    ROOT=.
    MEM_DIR=/tmp
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
T=200.0 # time to simulate
SEED=20 # pseudorandom number seed
SCALE=0.09 # scale of initial proposal relative to prior
SD=0.09 # adaptive proposal parameter (zero triggers default)
C=500 # no. samples to draw
A=4000 # no. steps before adaptation
MIN_TEMP=1.0 # minimum temperature (or temperature for single process)
MAX_TEMP=1.0 # maximum temperature
FILTER=ukf # filter type

# distributed MCMC settings
REMOTE=1 # 1 to enable remote proposals, 0 to disable
ALPHA=0.2 # remote proposal proportion
BETA=0.1 # remote proposal update propensity
R=50 # no. steps before incorporating remote proposal

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=256 # no. trajectories
L=0 # lookahead for auxiliary particle filter

# ODE settings
H=1.0
EPS_ABS=1.0e-6
EPS_REL=1.0e-3

# input files, in $DATA_DIR
INIT_FILE=C7_initHP2.nc # initial values file
FORCE_FILE=C7_force_pad.nc # forcings file
OBS_FILE=C7_S1_obs_padHP2.nc # observations file
PROPOSAL_FILE=C7_S1_proposal_padHP2.nc # proposal file
#INIT_FILE=OSP_71_76_init.nc # initial values file
#FORCE_FILE=OSP_71_76_force_pad.nc # forcings file
#OBS_FILE=OSP_71_76_obs_pad.nc # observations file
#PROPOSAL_FILE=OSP_71_76_proposal.nc # proposal file

# output file, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc

# intermediate results file 
FILTER_FILE=filter.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=2

# copy forcings and observations to memory file system, used repeatedly
mpirun -npernode 1 cp $DATA_DIR/$FORCE_FILE $DATA_DIR/$OBS_FILE $MEM_DIR/.

# output this script as record of settings
cat $ROOT/npzd/mcmc.sh

# run
mpirun -npernode $NPERNODE $ROOT/build/mcmc  --type $FILTER -P $P -T $T --min-temp $MIN_TEMP --max-temp $MAX_TEMP --sd $SD --scale $SCALE --remote $REMOTE --alpha $ALPHA --beta $BETA -R $R --min-ess $MIN_ESS --resampler $RESAMPLER -L $L -C $C -A $A -h $H --eps-abs $EPS_ABS --eps-rel $EPS_REL --seed $SEED --init-file $DATA_DIR/$INIT_FILE --force-file $MEM_DIR/$FORCE_FILE --obs-file $MEM_DIR/$OBS_FILE --filter-file $MEM_DIR/$FILTER_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $TMP_DIR/$OUTPUT_FILE --proposal-file $DATA_DIR/$PROPOSAL_FILE

# copy results from $TMP_DIR to $RESULTS_DIR
mpirun -npernode 1 sh -c "cp $TMP_DIR/$OUTPUT_FILE"'*'" $RESULTS_DIR/."
chmod 644 $RESULTS_DIR/*
