/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef USE_CPU

#include "model/NPZDModel.hpp"

#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/cuda/bind.hpp"
#include "bi/cuda/bind.cuh"

#include "bi/updater/SUpdater.hpp"
#include "bi/updater/DUpdater.hpp"
#include "bi/updater/CUpdater.hpp"
#include "bi/cuda/updater/SUpdater.cuh"
#include "bi/cuda/updater/DUpdater.cuh"
#include "bi/cuda/updater/CUpdater.cuh"

using namespace bi;

/*
 * Explicit class template instantiations.
 */
template class SUpdater<NPZDModel<> >;
template class DUpdater<NPZDModel<> >;
template class CUpdater<NPZDModel<> >;

#endif
