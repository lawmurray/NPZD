#!/bin/sh

##
## General settings
##

SUFFIX=   # suffix for output files
OUTPUT=1  # produce output?
TIME=1    # produce timings?

##
## Data settings
##

DATA_DIR=data        # directory containing data files
RESULTS_DIR=results  # directory to contain result files

INIT_FILENAME=C7_initHP2.nc   # init file
FORCE_FILENAME=C7_force.nc  # forcings file
OBS_FILENAME=C7_S1_obs_HP2.nc    # observations file

INIT_NS=0   # record along ns dimension to use for init file
FORCE_NS=0  # record along ns dimension to use for forcings file
OBS_NS=0    # record along ns dimension to use for observations file

##
## Simulation settings
##

T=365.0               # time to simulate
K=365                 # number of output points
P=1024                # number of particles
DELTA=1.0             # step size for random and discrete-time variables
H=1.0                 # step size for ODE integrator
ATOLER=1.0e-3         # absolute error tolerance for ODE integrator
RTOLER=1.0e-3         # relative error tolerance for ODE integrator
INCLUDE_PARAMETERS=1  # include  parameters as well as state?

##
## Prediction settings
##

U=730.0  # time to which to predict

##
## Particle filter settings
##

RESAMPLER=stratified  # resampling method
L=0                   # number of steps for Metropolis resampler
MIN_ESS=1.0           # minimum ESS to trigger resampling

##
## Kernel smoother settings
##

B=0 # kernel bandwidth (0 for default)

##
## PMCMC settings
##

SCALE=0.09  # scale of initial proposal relative to prior
SD=0.09     # adaptive proposal parameter (zero triggers default)
C=1024      # number of samples to draw
A=10000     # number of steps before starting adaptation

PROPOSAL_FILE=$RESULTS_DIR/urts${SUFFIX}.nc.$ID
FILTER_FILENAME=pf${SUFFIX}.nc.$ID

##
## Random number settings
##

SEED=0  # pseudorandom number seed

##
## System settings
## 

OMP_NUM_THREADS=4  # number of threads
