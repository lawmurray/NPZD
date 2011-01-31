/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/random/Random.hpp"
#include "bi/updater/RUpdater.hpp"
#include "bi/method/Simulator.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/buffer/SimulatorNetCDFBuffer.hpp"
#include "bi/misc/TicToc.hpp"

#include <iostream>
#include <string>
#include <unistd.h>
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
    ATOLER_ARG,
    RTOLER_ARG,
    NS_ARG,
    SEED_ARG,
    INIT_FILE_ARG,
    FORCE_FILE_ARG,
    OUTPUT_FILE_ARG,
    OUTPUT_ARG,
    TIME_ARG
  };
  real T = 0.0, H = 1.0, RTOLER = 1.0e-3, ATOLER = 1.0e-3;
  int P = 0, K = 0, NS = 0, SEED = 0;
  std::string INIT_FILE, FORCE_FILE, OUTPUT_FILE;
  bool OUTPUT = false, TIME = false;
  int c, option_index;

  option long_options[] = {
      {"atoler", required_argument, 0, ATOLER_ARG },
      {"rtoler", required_argument, 0, RTOLER_ARG },
      {"ns", required_argument, 0, NS_ARG },
      {"seed", required_argument, 0, SEED_ARG },
      {"init-file", required_argument, 0, INIT_FILE_ARG },
      {"force-file", required_argument, 0, FORCE_FILE_ARG },
      {"output-file", required_argument, 0, OUTPUT_FILE_ARG },
      {"output", required_argument, 0, OUTPUT_ARG },
      {"time", required_argument, 0, TIME_ARG }
  };
  const char* short_options = "T:P:K:h:";

  do {
    c = getopt_long(argc, argv, short_options, long_options, &option_index);
    switch(c) {
    case ATOLER_ARG:
      ATOLER = atof(optarg);
      break;
    case RTOLER_ARG:
      RTOLER = atof(optarg);
      break;
    case NS_ARG:
      NS = atoi(optarg);
      break;
    case SEED_ARG:
      SEED = atoi(optarg);
      break;
    case INIT_FILE_ARG:
      INIT_FILE = std::string(optarg);
      break;
    case FORCE_FILE_ARG:
      FORCE_FILE = std::string(optarg);
      break;
    case OUTPUT_FILE_ARG:
      OUTPUT_FILE = std::string(optarg);
      break;
    case OUTPUT_ARG:
      OUTPUT = atoi(optarg);
      break;
    case TIME_ARG:
      TIME = atoi(optarg);
      break;
    case 'T':
      T = atof(optarg);
      break;
    case 'P':
      P = atoi(optarg);
      break;
    case 'K':
      K = atoi(optarg);
      break;
    case 'h':
      H = atof(optarg);
      break;
    }
  } while (c != -1);

  /* bi init */
  #ifdef __CUDACC__
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

  /* state */
  Static<LOCATION> theta(m);
  State<LOCATION> s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, FORCE_FILE, NS);
  SparseInputNetCDFBuffer inInit(m, INIT_FILE, NS);

  /* initialise state from inputs */
  inInit.read(P_NODE, theta.get(P_NODE));
  inInit.read(D_NODE, s.get(D_NODE));
  inInit.read(C_NODE, s.get(C_NODE));

  /* output */
  SimulatorNetCDFBuffer* out;
  if (OUTPUT) {
    out = new SimulatorNetCDFBuffer(m, P, K, OUTPUT_FILE, NetCDFBuffer::REPLACE);
  } else {
    out = NULL;
  }

  /* simulate */
  RUpdater<NPZDModel<> > rUpdater(rng);
  BOOST_AUTO(sim, SimulatorFactory<LOCATION>::create(m, &rUpdater, &inForce, out));

  /* simulate */
  TicToc timer;
  sim->simulate(T, theta, s);

  /* output timing results */
  if (TIME) {
    synchronize();
    std::cout << timer.toc() << std::endl;
  }

  delete sim;
  delete out;
  return 0;
}
