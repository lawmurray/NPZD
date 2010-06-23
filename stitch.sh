#!/bin/bash
#PBS -l walltime=02:00:00,nodes=1:ppn=1,vmem=8gb
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    source init.sh
    ROOT=$PBS_O_WORKDIR
    INT_DIR=$MEMDIR
    ID=$PBS_JOBNAME
else
    ROOT=.
    INT_DIR=/tmp
    ID=stitch
fi
RESULTS_DIR=$ROOT/results

LD_LIBRARY_PATH=$ROOT/../bi/build:$LD_LIBRARY_PATH

# settings
SEED=3 # pseudorandom number seed
C=10000 # no. samples to draw
B=15000
I=20 # interval
INPUT_FILES=*-norep3* # input files, in $RESULTS_DIR
OUTPUT_FILE=$ID.nc.2 # output file, in $RESULTS_DIR

# output this script as record of settings
cat $ROOT/stitch.sh

# copy input files to local directory
cp $RESULTS_DIR/$INPUT_FILES $TMPDIR/.

# stitch
$ROOT/build/stitch -C $C -B $B -I $I --output-file $INT_DIR/$OUTPUT_FILE $TMPDIR/$INPUT_FILES 

# copy output file back to network dir
mv $INT_DIR/$OUTPUT_FILE $RESULTS_DIR/.

# clean up
rm $TMPDIR/$INPUT_FILES
