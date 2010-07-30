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
    ID=gibbs
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1 # no. OpenMP threads per process
NPERNODE=1 # no. processes per node

##
## Run config
##

# MCMC settings
T=365.0 # time to simulate
H=1.0 # initial step size
SEED=6 # pseudorandom number seed
SCALE=0.1 # scale of initial proposal relative to prior
ALPHA=0.0 # proportion of proposals to be non-local
SD=0.0 # adaptive proposal parameter (zero triggers default)
C=100 # no. samples to draw
A=1000 # no. steps before adaptation
MIN_TEMP=1.0 # minimum temperature (or temperature for single process)
MAX_TEMP=1.0 # maximum temperature

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=1024 # no. trajectories
L=10 # no. iterations for metropolis resampler

# input files, in $DATA_DIR
INIT_FILE=C7_init.nc # initial values file
FORCE_FILE=C7_force.nc # forcings file
OBS_FILE=C7_S1_obs_HP2.nc # observations file

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
#cat $ROOT/npzd/mcmc.sh

#mpirun -npernode $NPERNODE
$ROOT/build/gibbs -P $P -T $T -h $H --min-temp $MIN_TEMP --max-temp $MAX_TEMP --alpha $ALPHA --sd $SD --scale $SCALE --min-ess $MIN_ESS --resampler $RESAMPLER -L $L -C $C -A $A --seed $SEED --init-file $DATA_DIR/$INIT_FILE --force-file $TMP_DIR/$FORCE_FILE --obs-file $TMP_DIR/$OBS_FILE --filter-file $FILTER_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $RESULTS_DIR/$OUTPUT_FILE
