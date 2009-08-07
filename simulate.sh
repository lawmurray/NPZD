#!/bin/sh

P=10
K=365
T=365.0
SEED=0
OUTPUT="--output 1 --time 0"

build/simulate --seed $SEED $OUTPUT -P $P -K $K $B -T $T > results/simulate.csv
