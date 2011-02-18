#!/bin/bash
#PBS -l walltime=02:00:00,nodes=1:ppn=2,vmem=8gb
#PBS -j oe
#PBS -t 0-19

cd $PBS_O_WORKDIR

module load octave/3.3.51-gnu

octave src/converge.m $PBS_ARRAYID