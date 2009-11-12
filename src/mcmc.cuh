/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#ifndef MCMC_CUH
#define MCMC_CUH

#include "prior.hpp"
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

//#include "bi/method/ParticleMCMC.cuh"
#include "bi/pdf/ConditionalFactoredPdf.hpp"

bi::ParticleMCMC<NPZDModel,NPZDPrior,bi::ConditionalFactoredPdf<GET_TYPELIST(proposalP)> >* mcmc;

#endif
