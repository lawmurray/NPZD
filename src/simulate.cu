/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/method/Simulator.cuh"
#include "bi/cuda/method/Sampler.cuh"
#include "bi/cuda/model/BayesNet.cuh"

typedef bi::BayesNet<
    GET_TYPELIST(NPZDSTypeList),
    GET_TYPELIST(NPZDDTypeList),
    GET_TYPELIST(NPZDCTypeList),
    GET_TYPELIST(NPZDRTypeList),
    GET_TYPELIST(NPZDFTypeList),
    GET_TYPELIST(NPZDOTypeList),
    GET_TYPELIST(NPZDPTypeList)> NPZDBayesNet;

template class bi::Simulator<NPZDModel>;
template class bi::Sampler<NPZDModel>;
template void bi::model_init<NPZDBayesNet>(model&, const NPZDBayesNet&);
