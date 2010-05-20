/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef USE_CPU

#include "model/NPZDModel.hpp"

#include "bi/cuda/method/Simulator.cuh"
#include "bi/cuda/method/Sampler.cuh"

template class bi::Simulator<NPZDModel<> >;
template class bi::Sampler<NPZDModel<> >;

#endif
