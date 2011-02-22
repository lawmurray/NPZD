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
    NAME=urts
fi
RESULTS_DIR=$ROOT/results
DATA_DIR=$ROOT/data

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=4

##
## Run config
##

SEED=3 # pseudorandom number seed
OUTPUT=1 # produce output?
TIME=1 # produce timings?
INPUT_FILE=$RESULTS_DIR/ukf.nc # UKF results file
OUTPUT_FILE=$RESULTS_DIR/$NAME.nc # output file

$ROOT/build/urts --seed=$SEED --input-file=$INPUT_FILE --output-file=$OUTPUT_FILE --output=$OUTPUT --time=$TIME
