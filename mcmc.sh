#!/bin/bash
#PBS -l walltime=24:00:00,nodes=1:ppn=8,vmem=8gb
#PBS -j oe

#source /home/mur387/init.sh

ROOT=/home/lawrence/work/workspace
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

T=415.0 # time to simulate
H=1.0 # initial step size
SEED=3 # pseudorandom number seed
SCALE=0.01 # scale of initial proposal relative to prior
ALPHA=0.0 # proportion of proposals that are non-local
SD=0.0 # adaptive proposal parameter (zero triggers default)

B=0 # burn in steps
I=1 # sample interval (in steps)
C=40 # no. samples to draw
A=50 # no. steps before adaptation
MIN_TEMP=1.0 # minimum temperature (or temperature for single process)
MAX_TEMP=1.0 # maximum termperature

# particle filter settings
RESAMPLER=stratified # resampler to use, 'stratified' or 'metropolis'
MIN_ESS=1.0 # minimum ESS to trigger resampling (not used presently)
P=1024 # no. trajectories
L=10 # no. iterations for metropolis resampler

# job settings
ID=0 # job id
NPROCS=1 # no. processes

INIT_FILE=$ROOT/npzd/data/C7_initHP2.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/C7_force_pad.nc # forcings file
OBS_FILE=$ROOT/npzd/data/C7_S1_obs_pad.nc # observations file
FILTER_FILE=/tmp/pf.nc # intermediate results file
OUTPUT_FILE=$ROOT/npzd/results/mcmc.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

# output this script
#cat $ROOT/npzd/mcmc.sh

export OMP_NUM_THREADS=2 # no. OpenMP threads per process

#mpirun -np $NPROCS 
$ROOT/npzd/build/mcmc -P $P -T $T -h $H --min-temp $MIN_TEMP --max-temp $MAX_TEMP --alpha $ALPHA --sd $SD --scale $SCALE --min-ess $MIN_ESS --resampler $RESAMPLER -L $L -C $C -A $A --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --filter-file $FILTER_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE
