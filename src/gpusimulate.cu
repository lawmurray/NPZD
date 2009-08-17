/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "model/NPZDModel.cuh"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/method/MultiSimulator.cuh"
#include "bi/method/RUpdater.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/io/NetCDFReader.hpp"
#include "bi/io/NetCDFWriter.cuh"
#include "bi/random/Random.hpp"

#include <string>
#include "sys/time.h"

using namespace bi;

void simulate(const unsigned P, const unsigned K, const real_t T,
    const unsigned NS, const int SEED, const std::string& INPUT_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME) {
  //unsigned p;
  timeval start, end;

  /* random number generator */
  //Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  NcError ncErr(NcError::verbose_nonfatal);

  /* model */
  NPZDModel m;

  /* states */
  State s(m, P);
  Result r(m, P, K);

  /* initialise state */
  NetCDFReader<true,true,false,false> in(m, INPUT_FILE);
  in.read(s, NS);

  /* simulator */
  RUpdater<NPZDModel> rUpdater(s, SEED);
  FUpdater fUpdater(m, INPUT_FILE, s, NS);
  MultiSimulator<NPZDModel,real_t> sim(m, s, &r, &rUpdater, &fUpdater);

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));

  /* initial values of ex-nodes (N, P, Z & D) */
  //(prefer input file)
  //for (p = 0; p < P; ++p) {
  //  s.setExState(p, m.N, rng.uniform(1.0, 10.0));
  //  s.setExState(p, m.P, rng.uniform(1.0, 10.0));
  //  s.setExState(p, m.Z, rng.uniform(1.0, 10.0));
  //  s.setExState(p, m.D, rng.uniform(1.0, 10.0));
  //}

  /* simulate */
  gettimeofday(&start, NULL);
  sim.stepTo(T);
  gettimeofday(&end, NULL);

  /* output results */
  if (OUTPUT) {
    NetCDFWriter out(m, OUTPUT_FILE, P, K);
    out.write(r);
    out.write(s);
  }

  /* output timing */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }
}
