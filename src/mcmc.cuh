/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#ifndef MCMC_CUH
#define MCMC_CUH

#include "model/NPZDModel.hpp"

#include "bi/method/ParticleFilter.cuh"

bi::ParticleFilter<NPZDModel,real_t>* pf;

#endif
