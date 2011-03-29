/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "device.hpp"
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/pdf/AdditiveExpGaussianPdf.hpp"
#include "bi/pdf/ExpGaussianMixturePdf.hpp"

#include "bi/method/ParticleMCMC.hpp"
#include "bi/method/ParticleFilter.hpp"
#include "bi/method/AuxiliaryParticleFilter.hpp"
#include "bi/method/DisturbanceParticleFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MultinomialResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"

#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/AuxiliaryParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedRTSSmootherNetCDFBuffer.hpp"

#include "boost/typeof/typeof.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <getopt.h>

#ifdef USE_CPU
#define LOCATION ON_HOST
#else
#define LOCATION ON_DEVICE
#endif

using namespace bi;

int main(int argc, char* argv[]) {
  /* command line arguments */
  enum {
    ID_ARG,
    ATOLER_ARG,
    RTOLER_ARG,
    SCALE_ARG,
    SD_ARG,
    INIT_NS_ARG,
    FORCE_NS_ARG,
    OBS_NS_ARG,
    SEED_ARG,
    INIT_FILE_ARG,
    FORCE_FILE_ARG,
    OBS_FILE_ARG,
    OUTPUT_FILE_ARG,
    FILTER_FILE_ARG,
    PROPOSAL_FILE_ARG,
    RESAMPLER_ARG
  };
  real T = 0.0, H = 1.0, RTOLER = 1.0e-3, ATOLER = 1.0e-3;
  int ID = 0, P = 1024, INIT_NS = 0, FORCE_NS = 0, OBS_NS = 0,
      SEED = 0, C = 1000, M = 10;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, RESAMPLER = std::string("stratified");
  int c, option_index;

  option long_options[] = {
      {"id", required_argument, 0, ID_ARG },
      {"atoler", required_argument, 0, ATOLER_ARG },
      {"rtoler", required_argument, 0, RTOLER_ARG },
      {"init-ns", required_argument, 0, INIT_NS_ARG },
      {"force-ns", required_argument, 0, FORCE_NS_ARG },
      {"obs-ns", required_argument, 0, OBS_NS_ARG },
      {"seed", required_argument, 0, SEED_ARG },
      {"init-file", optional_argument, 0, INIT_FILE_ARG },
      {"force-file", required_argument, 0, FORCE_FILE_ARG },
      {"obs-file", required_argument, 0, OBS_FILE_ARG },
      {"output-file", required_argument, 0, OUTPUT_FILE_ARG },
      {"filter-file", required_argument, 0, FILTER_FILE_ARG },
      {"proposal-file", required_argument, 0, PROPOSAL_FILE_ARG },
      {"resampler", required_argument, 0, RESAMPLER_ARG }
  };
  const char* short_options = "T:h:P:C:M:";

  do {
    c = getopt_long(argc, argv, short_options, long_options, &option_index);
    switch(c) {
    case ID_ARG:
      ID = atoi(optarg);
      break;
    case ATOLER_ARG:
      ATOLER = atof(optarg);
      break;
    case RTOLER_ARG:
      RTOLER = atof(optarg);
      break;
    case INIT_NS_ARG:
      INIT_NS = atoi(optarg);
      break;
    case FORCE_NS_ARG:
      FORCE_NS = atoi(optarg);
      break;
    case OBS_NS_ARG:
      OBS_NS = atoi(optarg);
      break;
    case SEED_ARG:
      SEED = atoi(optarg);
      break;
    case INIT_FILE_ARG:
      if (optarg) {
        INIT_FILE = std::string(optarg);
      }
      break;
    case FORCE_FILE_ARG:
      FORCE_FILE = std::string(optarg);
      break;
    case OBS_FILE_ARG:
      OBS_FILE = std::string(optarg);
      break;
    case OUTPUT_FILE_ARG:
      OUTPUT_FILE = std::string(optarg);
      break;
    case FILTER_FILE_ARG:
      FILTER_FILE = std::string(optarg);
      break;
    case PROPOSAL_FILE_ARG:
      PROPOSAL_FILE = std::string(optarg);
      break;
    case RESAMPLER_ARG:
      RESAMPLER = std::string(optarg);
      break;
    case 'T':
      T = atof(optarg);
      break;
    case 'h':
      H = atof(optarg);
      break;
    case 'P':
      P = atoi(optarg);
      break;
    case 'C':
      C = atoi(optarg);
      break;
    case 'M':
      M = atoi(optarg);
      break;
    }
  } while (c != -1);

  /* bi init */
  #ifdef __CUDACC__
  int dev = chooseDevice(ID);
  std::cerr << "Using device " << dev << std::endl;
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  #endif
  bi_omp_init();
  bi_ode_init(H, ATOLER, RTOLER);
  h_ode_set_nsteps(100u);

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;
  const int ND = m.getNetSize(D_NODE);
  const int NC = m.getNetSize(C_NODE);
  const int NP = m.getNetSize(P_NODE);

  /* state and intermediate results */
  Static<LOCATION> theta(m);
  State<LOCATION> s(m, P);

  /* inputs and outputs */
  SparseInputNetCDFBuffer inForce(m, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, INIT_FILE, INIT_NS);
  UnscentedRTSSmootherNetCDFBuffer inProposal(m, PROPOSAL_FILE, NetCDFBuffer::READ_ONLY, STATIC_OWN);
  const int Y = inObs.countUniqueTimes(T);
  ParticleMCMCNetCDFBuffer out(m, C, Y, OUTPUT_FILE, NetCDFBuffer::REPLACE);
  AuxiliaryParticleFilterNetCDFBuffer tmp(m, P, Y, FILTER_FILE, NetCDFBuffer::REPLACE);

  /* set up resampler, filter and MCMC */
  StratifiedResampler resam(rng);
  //MultinomialResampler resam(rng);
  //MetropolisResampler resam(rng, 5);
  BOOST_AUTO(filter, ParticleFilterFactory<LOCATION>::create(m, rng, &inForce, &inObs, &tmp));
  BOOST_AUTO(mcmc, ParticleMCMCFactory<LOCATION>::create(m, rng, &out, INITIAL_CONDITIONED));

  /* initialise state */
  BOOST_AUTO(p0, mcmc->getPrior());
  ExpGaussianPdf<> q(p0.size());
  host_vector<real> x(p0.size());

  inProposal.readSmoothState(0, q.mean(), q.cov());

  q.addLogs(m.getPrior(D_NODE).getLogs(), 0);
  q.addLogs(m.getPrior(C_NODE).getLogs(), ND);
  q.addLogs(m.getPrior(P_NODE).getLogs(), ND + NC);
  q.init();

  //p0.sample(rng, x);
  q.sample(rng, x);

//  inInit.read(D_NODE, vector_as_row_matrix(subrange(x, 0, ND)));
//  inInit.read(C_NODE, vector_as_row_matrix(subrange(x, ND, NC)));
//  inInit.read(P_NODE, vector_as_row_matrix(subrange(x, ND + NC, NP)));

  mcmc->init(x, T, theta, s, filter, &resam);
  for (c = 0; c < C; ++c) {
    if ((c % M) == 0) {
      //p0.sample(rng, x);
      q.sample(rng, x);
    }
    mcmc->proposal(x);
    mcmc->prior();
    mcmc->likelihood(T, theta, s, filter, &resam);
    mcmc->accept(filter);
    mcmc->output(c);
    mcmc->report(c);
  }
  mcmc->term(theta);

  delete mcmc;
  delete filter;

  return 0;
}
