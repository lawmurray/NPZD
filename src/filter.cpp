/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "prior.hpp"
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/method/ParticleFilter.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/RUpdater.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/ForwardNetCDFReader.hpp"
#include "bi/io/ForwardNetCDFWriter.hpp"

#include "cuda_runtime.h"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* handle command line arguments */
  real_t T, H, MIN_ESS;
  unsigned P, K, INIT_NS, FORCE_NS, OBS_NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. particles")
    (",K", po::value(&K), "size of intermediate result buffer")
    (",T", po::value(&T), "simulation time for each trajectory")
    (",h", po::value(&H)->default_value(0.2),
        "suggested first step size for each trajectory")
    ("min-ess", po::value(&MIN_ESS)->default_value(1.0),
        "minimum ESS (as proportion of P) at each step to avoid resampling")
    ("seed", po::value(&SEED)->default_value(0),
        "pseudorandom number seed")
    ("init-file", po::value(&INIT_FILE),
        "input file containing initial values")
    ("force-file", po::value(&FORCE_FILE),
        "input file containing forcings")
    ("obs-file", po::value(&OBS_FILE),
        "input file containing observations")
    ("output-file", po::value(&OUTPUT_FILE),
        "output file to contain results")
    ("init-ns", po::value(&INIT_NS)->default_value(0),
        "index along ns dimension of initial value file to use")
    ("force-ns", po::value(&FORCE_NS)->default_value(0),
        "index along ns dimension of forcings file to use")
    ("obs-ns", po::value(&OBS_NS)->default_value(0),
        "index along ns dimension of observations file to use")
    ("output", po::value(&OUTPUT)->default_value(false), "enable output")
    ("time", po::value(&TIME)->default_value(false), "enable timing output");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 0;
  }

  /* report missing variables in NetCDF, but don't die */
  #ifndef NDEBUG
  NcError ncErr(NcError::verbose_nonfatal);
  #else
  NcError ncErr(NcError::silent_nonfatal);
  #endif

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(H));
  ode_set_rtoler(CUDA_REAL(1.0e-9));
  ode_set_atoler(CUDA_REAL(1.0e-9));
  ode_set_nsteps(1000);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* prior */
  NPZDPrior prior;

  /* state and intermediate results */
  State s(m, P);

  /* initialise from file... */
  ForwardNetCDFReader<true,true,true,false,true,true,true> in(m, INIT_FILE, INIT_NS);
  in.read(s);

  /* ...and/or initialise from prior */
  prior.getDPrior().sample(rng, s.dState);
  prior.getCPrior().sample(rng, s.cState);

  /* forcings, observations */
  FUpdater fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OUpdater oUpdater(m, OBS_FILE, s, OBS_NS);

  /* outputs */
  ForwardNetCDFWriter* out;
  if (OUTPUT) {
    out = new ForwardNetCDFWriter(m, OUTPUT_FILE, P, oUpdater.numUniqueTimes());
  } else {
    out = NULL;
  }
  std::ofstream essOut("ess.txt");

  /* filter */
  timeval start, end;
  gettimeofday(&start, NULL);

  unsigned t;
  real_t ess;
  ParticleFilter<NPZDModel> pf(m, s, rng, &fUpdater, &oUpdater);

  /* filter */
  cudaStream_t stream;
  CUDA_CHECKED_CALL(cudaStreamCreate(&stream));
  pf.bind(stream);
  pf.upload(stream);
  while (pf.getTime() < T) {
    BI_LOG("t = " << pf.getTime());
    pf.filter(T, stream);
    pf.weight(stream);
    ess = pf.ess(stream);
    essOut << ess << std::endl;
    pf.resample(MIN_ESS*s.P, stream);
    if (out != NULL) {
      pf.download(stream);
      cudaStreamSynchronize(stream);
      out->write(s, pf.getTime());
    }
  }
  cudaStreamSynchronize(stream);
  pf.unbind();
  CUDA_CHECKED_CALL(cudaStreamDestroy(stream));

  /* wrap up timing */
  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  return 0;
}
