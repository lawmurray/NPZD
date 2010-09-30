#!/bin/bash
#PBS -l walltime=00:30:00,nodes=4:ppn=8,vmem=8gb
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    source init.sh
    ROOT=$PBS_O_WORKDIR
    TMP_DIR=$MEMDIR
    ID=$PBS_JOBNAME
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
T=1875.0 # time to simulate
H=1.0 # initial step size
SEED=20 # pseudorandom number seed
SCALE=0.05 # scale of initial proposal relative to prior
ALPHA=0.2 # proportion of proposals to be non-local
SD=0.09 # adaptive proposal parameter (zero triggers default)
C=100 # no. samples to draw
A=10000 # no. steps before adaptation
MIN_TEMP=1.0 # minimum temperature (or temperature for single process)
MAX_TEMP=1.0 # maximum temperature
FILTER=ukf # filter type

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=200 # no. trajectories
L=0 # lookahead for auxiliary particle filter

# ODE settings
H=1.0
EPS_ABS=1.0e-6
EPS_REL=1.0e-3

# input files, in $DATA_DIR
#INIT_FILE=PZtest1_init.nc # initial values file
#FORCE_FILE=C7_force.nc # forcings file
#OBS_FILE=PZtest1_obs_gaussian.nc # observations file
INIT_FILE=C7_init.nc # initial values file
FORCE_FILE=OSP_71_76_force_pad.nc # forcings file
OBS_FILE=OSP_71_76_obs_pad.nc # observations file
PROPOSAL_FILE=OSP_71_76_proposal.nc # proposal file

# output file, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc

# intermediate results file 
FILTER_FILE=$TMP_DIR/pf.nc

# records to use from input files
INIT_NS=0
FORCE_NS=0
OBS_NS=0

# copy forcings and observations to local directory, used repeatedly
mpirun -npernode 1 cp $DATA_DIR/$FORCE_FILE $DATA_DIR/$OBS_FILE $TMP_DIR/.

# output this script as record of settings
#cat $ROOT/npzd/mcmc.sh

#mpirun -npernode $NPERNODE
$ROOT/build/mcmc  --type $FILTER -P $P -T $T --min-temp $MIN_TEMP --max-temp $MAX_TEMP --alpha $ALPHA --sd $SD --scale $SCALE --min-ess $MIN_ESS --resampler $RESAMPLER -L $L -C $C -A $A -h $H --eps-abs $EPS_ABS --eps-rel $EPS_REL --seed $SEED --init-file $DATA_DIR/$INIT_FILE --force-file $TMP_DIR/$FORCE_FILE --obs-file $TMP_DIR/$OBS_FILE --proposal-file $DATA_DIR/$PROPOSAL_FILE --filter-file $FILTER_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $RESULTS_DIR/$OUTPUT_FILE
