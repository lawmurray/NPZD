#!/bin/sh

bi simulate @simulate_osp.conf @config.conf --transform-param-to-state --seed 0 --enable-cuda
bi sample @sample_osp.conf @config.conf --seed 1
bi simulate @predict_osp.conf @config.conf --transform-param-to-state --init-file results/sample_osp.nc --seed 2 --enable-cuda
