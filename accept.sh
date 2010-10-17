#!/bin/bash
#PBS -l walltime=00:05:00,nodes=1:ppn=2,vmem=8gb
#PBS -j oe
#PBS -t 0-9

cd $PBS_O_WORKDIR

module load octave/3.3.51-gnu

octave src/accept.m $PBS_ARRAYID
