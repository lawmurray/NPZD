#include "model/NPZDModel.cuh"

#include "bi/cuda/cuda.hpp"
#include "bi/method/State.cuh"
#include "bi/method/Result.cuh"
#include "bi/method/FUpdater.cuh"
#include "bi/method/MultiSimulator.cuh"

#include <iostream>
#include <iomanip>
#include "sys/time.h"

using namespace bi;

int simulate(const unsigned P, const unsigned K,
    const real_t T, const bool write) {
  /* construct model */
  unsigned i, j, k;
  real_t x;
  timeval start, end;

  NPZDModel m;
  State s(m, P);
  Result r(m, P, K);
  FUpdater<NPZDModel> fUpdater(m, s.fState, "/mnt/data/data/CF1.nc");
  MultiSimulator<NPZDModel,real_t> sim(m, s, &r, &fUpdater);

  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(1000);

  gettimeofday(&start, NULL);

  /* initialise */
  for (j = 0; j < m.getExSize(); ++j) {
    for (i = 0; i < P; ++i) {
      x = (real_t)((real_t)rand() / (real_t)RAND_MAX);
      s.setExState(i,j,x);
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
        for (j = 0; j < m.getExSize(); ++j) {
          std::cout << '\t' << r.getExResult(i,j,k);
        }
        std::cout << std::endl;
      }

      /* final result */
      std::cout << T;
      for (j = 0; j < m.getExSize(); ++j) {
        std::cout << '\t' << s.getExState(i,j);
      }
      std::cout << std::endl << std::endl;
    }
  }

  gettimeofday(&end, NULL);

  return (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
}
