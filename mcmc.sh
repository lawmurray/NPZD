#!/bin/bash
#PBS -l walltime=24:00:00,nodes=2:ppn=1,gres=gpu,vmem=8gb
#PBS -j oe

source /home/mur387/init.sh

ROOT=/home/mur387
LD_LIBRARY_PATH=$ROOT/bi/build:$LD_LIBRARY_PATH

P=1024 # no. trajectories
T=200.0 # time to simulate
H=0.3 # initial step size
SCALE=0.1 # scale of initial proposal relative to prior
B=0 # burn in steps
I=1 # sample interval (in steps)
L=30000 # no. samples to draw
A=5000 # no. steps before adaptation
MIN_TEMP=1.0
MAX_TEMP=2.0
ALPHA=0.1
SD=0.02
MIN_ESS=1.0
ID=$PBS_JOBID

NPROCS=4
SEED=14756 # pseudorandom number seed

INIT_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # initial values file
FORCE_FILE=$ROOT/npzd/data/GPUinput_OSP_0D.nc # forcings file
OBS_FILE=$ROOT/npzd/data/GPUobs_EQ.nc # observations file
OUTPUT_FILE=$ROOT/npzd/results/mcmc-$ID.nc # output file

INIT_NS=0
FORCE_NS=0
OBS_NS=1

# output this script
cat $ROOT/npzd/mcmc.sh

time mpirun -np $NPROCS $ROOT/npzd/build/mcmc -P $P -T $T -h $H --min-temp $MIN_TEMP --max-temp $MAX_TEMP --alpha $ALPHA --sd $SD --scale $SCALE --min-ess $MIN_ESS -B $B -I $I -L $L -A $A --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE
