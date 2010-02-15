/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "model/NPZDModel.hpp"
#include "bi/state/State.hpp"

#include "bi/cuda/model/BayesNet.cuh"
#include "bi/cuda/method/ParticleFilter.cuh"
#include "bi/cuda/method/UnscentedKalmanFilter.cuh"

typedef bi::BayesNet<
    GET_TYPELIST(NPZDSTypeList),
    GET_TYPELIST(NPZDDTypeList),
    GET_TYPELIST(NPZDCTypeList),
    GET_TYPELIST(NPZDRTypeList),
    GET_TYPELIST(NPZDFTypeList),
    GET_TYPELIST(NPZDOTypeList),
    GET_TYPELIST(NPZDPTypeList)> NPZDBayesNet;

template void bi::model_init<NPZDBayesNet>(model&, const NPZDBayesNet&);
template class bi::ParticleFilter<NPZDModel>;
template class bi::UnscentedKalmanFilter<NPZDModel>;
