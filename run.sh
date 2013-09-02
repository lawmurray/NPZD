#!/bin/sh

libbi sample @config.conf @prior_osp.conf
libbi sample @config.conf @posterior_osp.conf
libbi sample @config.conf @prediction_osp.conf
