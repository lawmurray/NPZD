/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFWriter.hpp"
#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/method/ParticleFilter.cuh"

#include "boost/typeof/typeof.hpp"

#include <string>
#include "sys/time.h"

using namespace bi;

void filter(const unsigned P, const unsigned K, const real_t T,
    const unsigned NS, const int SEED, const std::string& INIT_FILE,
    const std::string& FORCE_FILE, const std::string& OBS_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME)  {
  /* random number generator */
  Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  NcError ncErr(NcError::verbose_nonfatal);

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(CUDA_REAL(1000));

  /* model */
  NPZDModel m;

  /* output */
  NetCDFWriter<real_t,true,true,true,true,true,true,true> out(m, OUTPUT_FILE, P, 365);

  /* state */
  State s(m, P);
  Result<real_t> r(m, P, K);

  /* initialise from file at this stage */
  NetCDFReader<real_t,true,true,true,false,false,false,true> in(m, INIT_FILE, NS);
  in.read(s);

  /* particle filter */
  FUpdater<real_t> fUpdater(m, FORCE_FILE, s, NS);
  OUpdater<real_t> oUpdater(m, OBS_FILE, s, NS);
  ParticleFilter<NPZDModel,real_t> pf(m, s, rng, &r, &fUpdater, &oUpdater);

  /* timer */
  timeval start, end;

  /* stream */
  cudaStream_t stream;
  CUDA_CHECKED_CALL(cudaStreamCreate(&stream));

  /* filter */
  gettimeofday(&start, NULL);
  pf.bind(stream);
  pf.upload(stream);
  while (pf.getTime() < T) {
    std::cerr << pf.getTime() << ' ';
    pf.advance(T, stream);
    pf.weight(stream);
    pf.resample(stream);
    if (OUTPUT) {
      pf.download(stream);
      cudaStreamSynchronize(stream);
      out.write(s, pf.getTime());
    }
  }
  std::cerr << std::endl;
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }
}
