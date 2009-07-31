#include "model/NPZDModel.cuh"

#include "bi/cuda/cuda.hpp"
#include "bi/method/MultiSimulator.cuh"

#include <iostream>
#include <iomanip>
#include "sys/time.h"

int simulate(const unsigned P, const unsigned K,
    const real_t T, const bool write) {
  /* construct model */
  unsigned i, j, k;
  real_t x;
  timeval start, end;

  NPZDModel model;
  bi::MultiSimulator<NPZDModel,real_t> sim(&model, P, K);

  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(1000);

  gettimeofday(&start, NULL);

  /* initialise */
  for (j = 0; j < model.getInSize(); ++j) {
    for (i = 0; i < P; ++i) {
      x = (real_t)((real_t)rand() / (real_t)RAND_MAX);
      sim.setInState(i,j,x);
    }
  }
  for (j = 0; j < model.getExSize(); ++j) {
    for (i = 0; i < P; ++i) {
      x = (real_t)((real_t)rand() / (real_t)RAND_MAX);
      sim.setExState(i,j,x);
    }
  }

  /* simulate */
  sim.stepTo(T);

  /* output */
  if (write) {
    i = 0;
    std::cout << std::setprecision(10);
    for (i = 0; i < P; ++i) {
      for (k = 0; k < K; ++k) {
        std::cout << T*k/K;
        for (j = 0; j < model.getExSize(); ++j) {
          std::cout << '\t' << sim.getExResult(i,j,k);
        }
        std::cout << std::endl;
      }

      /* final result */
      std::cout << T;
      for (j = 0; j < model.getExSize(); ++j) {
        std::cout << '\t' << sim.getExState(i,j);
      }
      std::cout << std::endl << std::endl;
    }
  }

  gettimeofday(&end, NULL);

  return (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
}
