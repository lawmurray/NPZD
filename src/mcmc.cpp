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
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MultinomialResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"

#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"

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
  real T = 0.0, H = 1.0, RTOLER = 1.0e-3, ATOLER = 1.0e-3,
      SCALE = 0.01, SD = 0.0;
  int ID = 0, P = 1024, L = 10, INIT_NS = 0, FORCE_NS = 0, OBS_NS = 0,
      SEED = 0, C = 100, A = 1000;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, RESAMPLER = std::string("stratified");
  int c, option_index;

  option long_options[] = {
      {"id", required_argument, 0, ID_ARG },
      {"atoler", required_argument, 0, ATOLER_ARG },
      {"rtoler", required_argument, 0, RTOLER_ARG },
      {"sd", required_argument, 0, SD_ARG },
      {"scale", required_argument, 0, SCALE_ARG },
      {"init-ns", required_argument, 0, INIT_NS_ARG },
      {"force-ns", required_argument, 0, FORCE_NS_ARG },
      {"obs-ns", required_argument, 0, OBS_NS_ARG },
      {"seed", required_argument, 0, SEED_ARG },
      {"init-file", optional_argument, 0, INIT_FILE_ARG },
      {"force-file", required_argument, 0, FORCE_FILE_ARG },
      {"obs-file", required_argument, 0, OBS_FILE_ARG },
      {"filter-file", required_argument, 0, FILTER_FILE_ARG },
      {"proposal-file", optional_argument, 0, PROPOSAL_FILE_ARG },
      {"output-file", required_argument, 0, OUTPUT_FILE_ARG },
      {"resampler", required_argument, 0, RESAMPLER_ARG }
  };
  const char* short_options = "T:h:P:L:C:A:";

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
    case SD_ARG:
      SD = atof(optarg);
      break;
    case SCALE_ARG:
      SCALE = atof(optarg);
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
      if (optarg) {
        PROPOSAL_FILE = std::string(optarg);
      }
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
    case 'L':
      L = atoi(optarg);
      break;
    case 'C':
      C = atoi(optarg);
      break;
    case 'A':
      A = atoi(optarg);
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

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;
  const int NP = m.getNetSize(P_NODE);
  const int ND = m.getNetSize(D_NODE);
  const int NC = m.getNetSize(C_NODE);
  if (SD <= 0.0) {
    SD = 2.4*2.4/NP;
  }

  /* state and intermediate results */
  Static<LOCATION> theta(m);
  State<LOCATION> s(m, P);

  /* inputs and outputs */
  SparseInputNetCDFBuffer inForce(m, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, INIT_FILE, INIT_NS);
  const int Y = inObs.countUniqueTimes(T);
  ParticleMCMCNetCDFBuffer out(m, C, Y, OUTPUT_FILE, NetCDFBuffer::REPLACE);
  ParticleFilterNetCDFBuffer outFilter(m, P, Y, FILTER_FILE, NetCDFBuffer::REPLACE);

  /* proposal distribution and Markov chain initialisation */
  AdditiveExpGaussianPdf<> q(NP);
  if (PROPOSAL_FILE.compare("") != 0) {
    /* construct from proposal file */
    /**
     * @todo Online UKF is run with a different model (p-nodes as d-nodes),
     * the approach here works only if ND + NC >= NP.
     */
    UnscentedKalmanFilterNetCDFBuffer inProposal(m, PROPOSAL_FILE);
    host_vector<real> mu(ND + NC);
    host_matrix<real> Sigma(ND + NC, ND + NC);

    inProposal.readCorrectedState(inProposal.size2() - 1, mu, Sigma);
    int j;
    for (j = 0; j < Sigma.size2(); ++j) {
      scal(SD, column(Sigma, j));
    }
    q.setCov(subrange(Sigma, 0, NP, 0, NP));

    /* initialise chain from mean of online posterior... */
    //expVec(mu, m.getPrior(P_NODE).getLogs());
    //row(theta.get(P_NODE), 0) = subrange(mu, 0, NP);

    /* ...or sample from online posterior */
    m.getPrior(P_NODE).samples(rng, theta.get(P_NODE));
  } else {
    /* construct from scaled prior */
    host_matrix<real> Sigma(NP,NP);
    Sigma = m.getPrior(P_NODE).cov();

    int j;
    for (j = 0; j < Sigma.size2(); ++j) {
      scal(SCALE, column(Sigma, j));
    }
    //////// add next two lines for PZ model ////////
    //diagonal(Sigma)(0) = pow(0.01,2);
    //diagonal(Sigma)(1) = pow(0.005,2);

    q.setCov(Sigma);

    /* initialise chain from file... */
    //inInit.read(theta.get(P_NODE));

    /* ...or prior... */
    m.getPrior(P_NODE).samples(rng, theta.get(P_NODE));

    /* ...or special case */
    //////// add next two lines for PZ model ////////
    //s.get(P_NODE)(0,0) = 0.2;
    //s.get(P_NODE)(0,1) = 0.15;
  }

  ///////// remove next line for PZ model ////////
  q.setLogs(m.getPrior(P_NODE).getLogs());

  /* filter */
  StratifiedResampler resam(rng);
  BOOST_AUTO(filter, ParticleFilterFactory<LOCATION>::create(m, rng, &inForce, &inObs, &outFilter));

  /* sampler */
  BOOST_AUTO(mcmc, ParticleMCMCFactory<LOCATION>::create(m, rng, &out));
  mcmc->sample(q, C, T, theta, s, filter, &resam, SD, A);

  std::cout << mcmc->getNumAccepted() << " of " << mcmc->getNumSteps() <<
      " proposals accepted" << std::endl;

  delete mcmc;
  delete filter;

  return 0;
}
