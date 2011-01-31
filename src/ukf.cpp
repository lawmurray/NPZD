/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/misc/TicToc.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <getopt.h>

using namespace bi;

int main(int argc, char* argv[]) {
  /* openmp */
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-6, 1.0e-3);
  h_ode_set_nsteps(100);

  /* command line arguments */
  enum {
    SEED_ARG,
    INIT_FILE_ARG,
    FORCE_FILE_ARG,
    OBS_FILE_ARG,
    OUTPUT_FILE_ARG,
    INIT_NS_ARG,
    FORCE_NS_ARG,
    OBS_NS_ARG,
    OUTPUT_ARG,
    TIME_ARG
  };
  int SEED = 0;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  int INIT_NS = 0, FORCE_NS = 0, OBS_NS = 0;
  bool OUTPUT = false, TIME = false;
  real T, h = 0.1;
  int c, option_index;

  option long_options[] = {
      {"seed", required_argument, 0, SEED_ARG },
      {"init-file", required_argument, 0, INIT_FILE_ARG },
      {"force-file", required_argument, 0, FORCE_FILE_ARG },
      {"obs-file", required_argument, 0, OBS_FILE_ARG },
      {"output-file", required_argument, 0, OUTPUT_FILE_ARG },
      {"init-ns", required_argument, 0, INIT_NS_ARG },
      {"force-ns", required_argument, 0, FORCE_NS_ARG },
      {"obs-ns", required_argument, 0, OBS_NS_ARG },
      {"output", required_argument, 0, OUTPUT_ARG },
      {"time", required_argument, 0, TIME_ARG }
  };
  const char* short_options = "T:h:";

  do {
    c = getopt_long(argc, argv, short_options, long_options, &option_index);
    switch(c) {
    case SEED_ARG:
      SEED = atoi(optarg);
      break;
    case INIT_FILE_ARG:
      INIT_FILE = std::string(optarg);
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
    case INIT_NS_ARG:
      INIT_NS = atoi(optarg);
      break;
    case FORCE_NS_ARG:
      FORCE_NS = atoi(optarg);
      break;
    case OBS_NS_ARG:
      OBS_NS = atoi(optarg);
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
    case 'h':
      h = atof(optarg);
      break;
    }
  } while (c != -1);

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* model */
  NPZDModel<> m;

  /* state and intermediate results */
  int P = 1;
  Static<ON_HOST> theta(m);
  State<ON_HOST> s(m, P);

  /* random number generator */
  Random rng(SEED);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, INIT_FILE, INIT_NS);

  /* initialise state */
  inInit.read(P_NODE, theta.get(P_NODE));

  /* output */
  UnscentedKalmanFilterNetCDFBuffer* out;
  if (OUTPUT) {
    out = new UnscentedKalmanFilterNetCDFBuffer(m, P, inObs.countUniqueTimes(T) + 1,
        OUTPUT_FILE, NetCDFBuffer::REPLACE);
  } else {
    out = NULL;
  }

  /* set filter */
  BOOST_AUTO(filter, createUnscentedKalmanFilter(m, rng, &inForce, &inObs, out));

  /* do filter */
  TicToc timer;
  filter->filter(T, theta, s);

  /* output timing results */
  if (TIME) {
    std::cout << timer.toc() << std::endl;
  }

  delete out;
  delete filter;

  return 0;
}
