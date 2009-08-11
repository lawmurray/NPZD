/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "model/NPZDModel.cuh"

#include "bi/cuda/cuda.hpp"
#include "bi/method/MultiSimulator.cuh"
#include "bi/io/NetCDFWriter.cuh"
#include "bi/random/Random.hpp"

#include <string>
#include "sys/time.h"

using namespace bi;

void simulate(const unsigned P, const unsigned K, const real_t T,
    const int SEED, const std::string& INPUT_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME) {
  unsigned p;
  timeval start, end;

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* states */
  State s(m, P);
  Result r(m, P, K);

  /* simulator */
  FUpdater<NPZDModel> fUpdater(m, s.fState, INPUT_FILE.c_str());
  MultiSimulator<NPZDModel,real_t> sim(m, s, &r, &fUpdater);

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));

  /* initial values of ex-nodes (N, P, Z & D) */
  for (p = 0; p < P; ++p) {
    s.setExState(p, m.N, rng.uniform(1.0, 10.0));
    s.setExState(p, m.P, rng.uniform(1.0, 10.0));
    s.setExState(p, m.Z, rng.uniform(1.0, 10.0));
    s.setExState(p, m.D, rng.uniform(1.0, 10.0));
  }

  /* simulate */
  gettimeofday(&start, NULL);
  sim.stepTo(T);
  gettimeofday(&end, NULL);

  /* output results */
  if (OUTPUT) {
    NetCDFWriter out(m, OUTPUT_FILE.c_str(), P, K);
    out.write(r);
    out.write(s);
  }

  /* output timing */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }
}
