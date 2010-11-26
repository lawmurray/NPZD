#!/bin/bash
#PBS -l walltime=04:00:00,nodes=1:ppn=4,vmem=16gb
#PBS -j oe

##
## System config
##
if [[ "`hostname`" == gpu* ]]
then
    module load octave/3.3.51-gnu
    ROOT=$PBS_O_WORKDIR
    NAME=$PBS_JOBNAME
    C=$PBS_ARRAYID
else
    ROOT=.
    NAME=haario
    C=4
fi
cd $ROOT

A=40 # number of ensembles (numbered 0 to A - 1 as array job)
S=25000 # number of steps in each chain

octave src/converge_across.m $NAME $A $C $S
