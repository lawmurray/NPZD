#!/bin/sh
#PBS -l walltime=24:00:00,nodes=1:ppn=4:gpus=1,vmem=64gb -j oe

source $HOME/init.sh
cd $PBS_O_WORKDIR

SEED=$PBS_ARRAYID
INIT_NP=$PBS_ARRAYID

libbi test_filter @config.conf @test_filter.conf --filter bootstrap --output-file results/test_exact-$INIT_NP.nc --init-np $INIT_NP --nparticles 1048576 --reps 1 --Ps 1 --enable-cuda
