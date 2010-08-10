/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#ifndef USE_CPU

#include "model/NPZDModel.hpp"

#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/StratifiedResampler.inl"
#include "bi/method/Resampler.inl"
#include "bi/cuda/method/Resampler.cuh"

#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/cuda/bind.hpp"
#include "bi/cuda/bind.cuh"

#include "bi/updater/SUpdater.hpp"
#include "bi/updater/DUpdater.hpp"
#include "bi/updater/CUpdater.hpp"
#include "bi/updater/LUpdater.hpp"
#include "bi/cuda/updater/SUpdater.cuh"
#include "bi/cuda/updater/DUpdater.cuh"
#include "bi/cuda/updater/CUpdater.cuh"
#include "bi/cuda/updater/LUpdater.cuh"

#include "bi/cuda/math/vector.hpp"
#include "bi/cuda/math/index.hpp"

using namespace bi;

/*
 * Explicit class template instantiations.
 */
template class SUpdater<NPZDModel<> >;
template class DUpdater<NPZDModel<> >;
template class CUpdater<NPZDModel<> >;
template class LUpdater<NPZDModel<> >;

/*
 * Explicit function template instantiations.
 */
typedef gpu_vector<> V1;
typedef gpu_index V2;

template void LUpdater<NPZDModel<> >::update<host_index,gpu_vector<> >(const host_index&, gpu_vector<>&);
template void StratifiedResampler::resample<V1,V2>(V1&, V2&);
template void StratifiedResampler::resample<V1,V2>(const V1&, V1&, V2&);
template void StratifiedResampler::resample<V1,V2>(const V2::value_type, V1&, V2&);
template void StratifiedResampler::resample<V1,V2>(const V2::value_type, const V1&, V1&, V2&);

#endif
