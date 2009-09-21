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
#include "bi/random/Random.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFReader.hpp"
#include "bi/io/NetCDFWriter.cuh"

#include <string>
#include "sys/time.h"

using namespace bi;

void simulate(const unsigned P, const unsigned K, const real_t T,
    const unsigned NS, const int SEED, const std::string& INIT_FILE,
    const std::string& FORCE_FILE, const std::string& OBS_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME) {
  /* random number generator */
  Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  NcError ncErr(NcError::verbose_nonfatal);

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));

  /* model */
  NPZDModel m;

  /* state */
  State s(m, P);
  NetCDFReader<real_t,true,true,true,false,false,false,true> in(m, INIT_FILE, NS);
  in.read(s); // initialise state

  /* intermediate result buffer */
  Result<real_t> r(m, P, K);

  /* output */
  NetCDFWriter<real_t> out(m, OUTPUT_FILE, P, 366);

  /* simulator */
  FUpdater<real_t> fUpdater(m, FORCE_FILE, s, NS);
  OUpdater<real_t> oUpdater(m, OBS_FILE, s, NS);
  MultiSimulator<NPZDModel,real_t> sim(m, s, rng, &r, &fUpdater, &oUpdater);

  /* simulate and output */
  timeval start, end;
  gettimeofday(&start, NULL);
  unsigned k;
  while (sim.getTime() < T) {
    k = sim.simulate(T);
    if (OUTPUT) {
      out.write(r, k);
    }
  }
  if (OUTPUT) {
    out.write(s, sim.getTime());
  }
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }
}
