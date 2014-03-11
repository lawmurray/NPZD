#!/bin/sh

libbi test_filter @config.conf @test_filter.conf --filer bootstrap --output-file results/test_bootstrap.nc
libbi test_filter @config.conf @test_filter.conf --filer bridge --output-file results/test_bridge.nc
