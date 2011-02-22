/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1251 $
 * $Date: 2011-01-31 18:40:46 +0800 (Mon, 31 Jan 2011) $
 */
#include "model/NPZDModel.hpp"

#include "bi/method/UnscentedRTSSmoother.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedRTSSmootherNetCDFBuffer.hpp"
#include "bi/misc/TicToc.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sys/time.h>
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
    SEED_ARG,
    INPUT_FILE_ARG,
    OUTPUT_FILE_ARG,
    OUTPUT_ARG,
    TIME_ARG,
    ESTIMATE_PARAMETERS_ARG
  };
  int SEED = 0;
  std::string INPUT_FILE, OUTPUT_FILE;
  bool OUTPUT = false, TIME = false, ESTIMATE_PARAMETERS = false;
  int c, option_index;

  option long_options[] = {
      {"seed", required_argument, 0, SEED_ARG },
      {"input-file", required_argument, 0, INPUT_FILE_ARG },
      {"output-file", required_argument, 0, OUTPUT_FILE_ARG },
      {"output", required_argument, 0, OUTPUT_ARG },
      {"time", required_argument, 0, TIME_ARG },
      {"estimate-parameters", required_argument, 0, ESTIMATE_PARAMETERS_ARG }
  };
  const char* short_options = "";

  do {
    c = getopt_long(argc, argv, short_options, long_options, &option_index);
    switch(c) {
    case SEED_ARG:
      SEED = atoi(optarg);
      break;
    case INPUT_FILE_ARG:
      INPUT_FILE = std::string(optarg);
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
    case ESTIMATE_PARAMETERS_ARG:
      ESTIMATE_PARAMETERS = atoi(optarg);
      break;
    }
  } while (c != -1);

  /* bi init */
  #ifdef __CUDACC__
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  #endif
  bi_omp_init();

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* model */
  NPZDModel<> m;

  /* random number generator */
  Random rng(SEED);

  /* inputs */
  UnscentedKalmanFilterNetCDFBuffer in(m, INPUT_FILE, NetCDFBuffer::READ_ONLY,
      ESTIMATE_PARAMETERS ? STATIC_OWN : STATIC_SHARED);

  /* output */
  UnscentedRTSSmootherNetCDFBuffer* out;
  if (OUTPUT) {
    out = new UnscentedRTSSmootherNetCDFBuffer(m, in.size2(),
        OUTPUT_FILE, NetCDFBuffer::REPLACE,
        ESTIMATE_PARAMETERS ? STATIC_OWN : STATIC_SHARED);
  } else {
    out = NULL;
  }

  /* smooth */
  TicToc timer;
  if (ESTIMATE_PARAMETERS) {
    BOOST_AUTO(smoother, (UnscentedRTSSmootherFactory<LOCATION,STATIC_OWN>::create(m, rng, out)));
    smoother->smooth(&in);
    delete smoother;
  } else {
    BOOST_AUTO(smoother, (UnscentedRTSSmootherFactory<LOCATION,STATIC_SHARED>::create(m, rng, out)));
    smoother->smooth(&in);
    delete smoother;
  }

  /* output timing results */
  if (TIME) {
    std::cout << timer.toc() << std::endl;
  }

  delete out;

  return 0;
}
