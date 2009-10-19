/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "mcmc.hpp"
#include "mcmc.cuh"

#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/method/ParticleFilter.cuh"

using namespace bi;

void init(const real_t h, NPZDModel& m, State& s, Random& rng,
    FUpdater<>* fUpdater, OUpdater<>* oUpdater) {
  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(h));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(200);

  pf = new ParticleFilter<NPZDModel,real_t>(m, s, rng, NULL, fUpdater, oUpdater);
}

real_t filter(const real_t T, const real_t minEss) {
  pf->reset();
  real_t l = pf->filter(T, minEss);

  return l;
}

void destroy() {
  delete pf;
}
