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
    ID=simulate
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=2

##
## Run config
##

P=32 # no. trajectories
K=365 # number of output points
T=365.0 # time to simulate
H=1.0 # initial step size for ODE integrator
ATOLER=1.0e-3 # absolute error tolerance for ODE integrator
RTOLER=1.0e-3 # relative error tolerance for ODE integrator
SEED=0 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=1 # produce timings?

INIT_FILE=$DATA_DIR/C7_initHP2.nc # initial values file
FORCE_FILE=$DATA_DIR/C7_force.nc # forcings file
OUTPUT_FILE=$RESULTS_DIR/$ID.nc # output file
NS=0 # record number in input files

$ROOT/build/simulate -h $H --atoler=$ATOLER --rtoler=$RTOLER --ns=$NS -P $P -K $K -T $T --seed=$SEED --init-file=$INIT_FILE --force-file=$FORCE_FILE --output-file=$OUTPUT_FILE --output=$OUTPUT --time=$TIME
