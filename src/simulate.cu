/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef USE_CPU

#include "model/NPZDModel.hpp"

#include "bi/method/Simulator.hpp"
#include "bi/method/Sampler.hpp"

template class bi::Simulator<NPZDModel<> >;
template class bi::Sampler<NPZDModel<> >;

#endif
