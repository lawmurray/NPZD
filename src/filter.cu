/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#ifndef USE_CPU

#include "model/NPZDModel.hpp"
#include "bi/state/State.hpp"
#include "bi/cuda/method/StratifiedResampler.cuh"
#include "bi/cuda/method/ParticleFilter.cuh"
//#include "bi/cuda/method/UnscentedKalmanFilter.cuh"

using namespace bi;

template class ParticleFilter<NPZDModel<>, StratifiedResampler>;
//template class UnscentedKalmanFilter<NPZDModel<> >;

#endif
