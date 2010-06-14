#!/bin/bash
#PBS -l walltime=00:30:00,nodes=1:ppn=1,vmem=8gb
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
    ID=stitch
fi
RESULTS_DIR=$ROOT/results

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

# settings
SEED=3 # pseudorandom number seed
C=10000 # no. samples to draw
I=100 # interval
INPUT_FILES=*-repeat* # input files, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc # output file, in $RESULTS_DIR

# output this script as record of settings
#cat $ROOT/stitch.sh

$ROOT/build/stitch -C $C -I $I --output-file $RESULTS_DIR/$OUTPUT_FILE $RESULTS_DIR/$INPUT_FILES 
