/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/method/ParticleFilter.cuh"
#include "bi/cuda/method/UnscentedKalmanFilter.cuh"

template class bi::ParticleFilter<NPZDModel>;
template class bi::UnscentedKalmanFilter<NPZDModel>;
