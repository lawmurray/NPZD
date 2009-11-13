/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "prior.hpp"
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/method/ParticleMCMC.cuh"

template class bi::ParticleMCMC<NPZDModel,NPZDPrior,bi::ConditionalFactoredPdf<GET_TYPELIST(proposalP)> >;
