/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "model/NPZDModel.cuh"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/method/ParticleFilter.cuh"
#include "bi/random/Random.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFWriter.cuh"
//#include "bi/pdf/LogNormalPdf.hpp"

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
  ode_set_nsteps(CUDA_REAL(100));

  /* model */
  NPZDModel m;

  /* output */
  NetCDFWriter<real_t> out(m, OUTPUT_FILE, P, 1826);

  /* state */
  State s(m, P);
  Result<real_t> r(m, P, K);

  /* priors */
  //BOOST_AUTO(buildPPrior(m), p0);
  //BOOST_AUTO(buildDPrior(m), d0);
  //BOOST_AUTO(buildCPrior(m), c0);

  //d0.sample(s.dState, rng);
  //c0.sample(s.cState, rng);

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
    pf.download(stream);
    cudaStreamSynchronize(stream);
    if (OUTPUT) {
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

//LogNormalPdf<vector,banded_matrix> buildPPrior(NPZDModel& m) {
//  const unsigned N = m.getPSize();
//
//  vector mu(N);
//  banded_matrix sigma(N,N);
//  BOOST_AUTO(sigmad, diag(sigma));
//
//  mu[m.getPNode("KW")->getId()] = 0.03;
//  mu[m.getPNode("KC")->getId()] = 0.04;
//  mu[m.getPNode("deltaS")->getId()] = 5.0; // should be Gaussian!
//  mu[m.getPNode("deltaI")->getId()] = 0.5;
//  mu[m.getPNode("P_DF")->getId()] = 0.4;
//  mu[m.getPNode("Z_DF")->getId()] = 0.4;
//  mu[m.getPNode("alphaC")->getId()] = 1.2;
//  mu[m.getPNode("alphaCN")->getId()] = 0.4;
//  mu[m.getPNode("alphaCh")->getId()] = 0.03;
//  mu[m.getPNode("alphaA")->getId()] = 0.3;
//  mu[m.getPNode("alphaNC")->getId()] = 0.25;
//  mu[m.getPNode("alphaI")->getId()] = 4.7;
//  mu[m.getPNode("alphaCl")->getId()] = 0.2;
//  mu[m.getPNode("alphaE")->getId()] = 0.32;
//  mu[m.getPNode("alphaR")->getId()] = 0.1;
//  mu[m.getPNode("alphaQ")->getId()] = 0.01;
//  mu[m.getPNode("alphaL")->getId()] = 0.0;
//
//  sigmad[m.getPNode("KW")->getId()] = 0.2;
//  sigmad[m.getPNode("KC")->getId()] = 0.3;
//  sigmad[m.getPNode("deltaS")->getId()] = 1.0;
//  sigmad[m.getPNode("deltaI")->getId()] = 0.1;
//  sigmad[m.getPNode("P_DF")->getId()] = 0.2;
//  sigmad[m.getPNode("Z_DF")->getId()] = 0.2;
//  sigmad[m.getPNode("alphaC")->getId()] = 0.63;
//  sigmad[m.getPNode("alphaCN")->getId()] = 0.2;
//  sigmad[m.getPNode("alphaCh")->getId()] = 0.37;
//  sigmad[m.getPNode("alphaA")->getId()] = 1.0;
//  sigmad[m.getPNode("alphaNC")->getId()] = 0.3;
//  sigmad[m.getPNode("alphaI")->getId()] = 0.7;
//  sigmad[m.getPNode("alphaCl")->getId()] = 1.3;
//  sigmad[m.getPNode("alphaE")->getId()] = 0.25;
//  sigmad[m.getPNode("alphaR")->getId()] = 0.5;
//  sigmad[m.getPNode("alphaQ")->getId()] = 1.0;
//  sigmad[m.getPNode("alphaL")->getId()] = 0.0;
//
//  return LogNormalPdf<vector,banded_matrix>(mu, sigma);
//}
