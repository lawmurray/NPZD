#!/bin/sh
#PBS -l walltime=08:00:00,nodes=1:ppn=8,vmem=32gb
#PBS -j oe

module load octave/3.3.51-gnu

cd $PBS_O_WORKDIR
octave --path octave --path ../octave --eval "prepare_converge()"
