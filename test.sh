#!/bin/sh

libbi test_filter @config.conf @test_filter_osp.conf --filter bootstrap > bootstrap.csv
libbi test_filter @config.conf @test_filter_osp.conf --filter bridge > bridge.csv
