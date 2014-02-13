#!/bin/sh

for N in 32 64 128 256 512 1024
do
    libbi test_filter @config.conf @test_filter_osp.conf --filter bootstrap --nparticles $N > bootstrap-$N.csv
    libbi test_filter @config.conf @test_filter_osp.conf --filter bridge --nparticles $N > bridge-$N.csv
done
