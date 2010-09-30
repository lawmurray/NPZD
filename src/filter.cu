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
#include "bi/method/MultinomialResampler.hpp"
#include "bi/method/MultinomialResampler.inl"
#include "bi/method/MetropolisResampler.hpp"
#include "bi/method/MetropolisResampler.inl"
#include "bi/cuda/method/MetropolisResampler.cuh"
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
#include "bi/updater/OUpdater.hpp"

#include "bi/cuda/updater/SUpdater.cuh"
#include "bi/cuda/updater/DUpdater.cuh"
#include "bi/cuda/updater/CUpdater.cuh"
#include "bi/cuda/updater/LUpdater.cuh"
#include "bi/cuda/updater/OUpdater.cuh"

using namespace bi;

/*
 * Explicit class template instantiations.
 */
template class SUpdater<NPZDModel<> >;
template class DUpdater<NPZDModel<> >;
template class CUpdater<NPZDModel<> >;
template class LUpdater<NPZDModel<> >;
template class OUpdater<NPZDModel<> >;

/*
 * Explicit function template instantiations.
 */
typedef gpu_vector<> V1;
typedef gpu_vector<int> V2;
typedef host_vector<real> V3;
typedef host_vector<real, pinned_allocator<real> > V4;
typedef host_vector<int> V5;

template void StratifiedResampler::resample<V1,V2>(V1&, V2&);
template void StratifiedResampler::resample<V1,V1,V2>(const V1&, V1&, V2&);
template void StratifiedResampler::resample<V1,V2>(const int, V1&, V2&);
template void StratifiedResampler::resample<V1,V1,V2>(const int, const V1&, V1&, V2&);

template void StratifiedResampler::resample<V4,V5>(V4&, V5&);
template void StratifiedResampler::resample<V3,V4,V5>(const V3&, V4&, V5&);
template void StratifiedResampler::resample<V4,V5>(const int, V4&, V5&);
template void StratifiedResampler::resample<V3,V4,V5>(const int, const V3&, V4&, V5&);

template void MultinomialResampler::resample<V4,V5>(V4&, V5&);
template void MultinomialResampler::resample<V3,V4,V5>(const V3&, V4&, V5&);
template void MultinomialResampler::resample<V4,V5>(const int, V4&, V5&);
template void MultinomialResampler::resample<V3,V4,V5>(const int, const V3&, V4&, V5&);

template void MetropolisResampler::resample<V4,V5>(V4&, V5&);
template void MetropolisResampler::resample<V3,V4,V5>(const V3&, V4&, V5&);
template void MetropolisResampler::resample<V4,V5>(const int, V4&, V5&);
template void MetropolisResampler::resample<V3,V4,V5>(const int, const V3&, V4&, V5&);

template void Resampler::copy<V5>(const V5&, State&);
template void LUpdater<NPZDModel<> >::update<V5,V4>(const V5&, V4&);
template void OUpdater<NPZDModel<> >::update<V2>(const V2&, const int);
template void OUpdater<NPZDModel<> >::update<V5>(const V5&, const int);

#endif
