/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "mcmc.cuh"

using namespace bi;

void init(NPZDModel& m, NPZDPrior& x0,
    ConditionalFactoredPdf<GET_TYPELIST(proposalP)>& q, State& s, Random& rng,
    FUpdater* fUpdater, OUpdater* oUpdater) {
  mcmc = new ParticleMCMC<NPZDModel,NPZDPrior,
      ConditionalFactoredPdf<GET_TYPELIST(proposalP)> >(m, x0, q, s, rng,
      fUpdater, oUpdater);
}

bool step(const real_t T, const real_t minEss, const double lambda,
    bi::state_vector& theta, double& l) {
  bool result;

  result = mcmc->step(T, minEss, lambda);
  theta = mcmc->getState();
  l = mcmc->getLogLikelihood();

  return result;
}

void destroy() {
  delete mcmc;
}
