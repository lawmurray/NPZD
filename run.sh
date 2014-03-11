#!/bin/sh

libbi sample @config.conf @prior.conf
libbi sample @config.conf @posterior.conf
libbi sample @config.conf @prediction.conf
