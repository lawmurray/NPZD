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

#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;

using namespace bi;

int main(int argc, char* argv[]) {
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

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(H);
  ode_set_rtoler(1.0e-3);
  ode_set_atoler(1.0e-3);
  ode_set_nsteps(200);

  /* random number generator */
  Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  #ifndef NDEBUG
  NcError ncErr(NcError::verbose_nonfatal);
  #else
  NcError ncErr(NcError::silent_nonfatal);
  #endif

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
  BOOST_AUTO(d0, prior.getDPrior());
  BOOST_AUTO(c0, prior.getCPrior());
  d0.sample(rng, s.dState);
  c0.sample(rng, s.cState);

  /* forcings, observations */
  FUpdater fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OUpdater oUpdater(m, OBS_FILE, s, OBS_NS);

  /* outputs */
  ForwardNetCDFWriter* out;
  if (OUTPUT) {
    out = new ForwardNetCDFWriter(m, OUTPUT_FILE, P, oUpdater.numUniqueTimes() + 1);
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
  for (t = 0; t < T; ++t) {
    pf.filter(t, MIN_ESS*P);
    //ess = pf.ess(stream);
    //essOut << ess << std::endl;
    out->write(s, pf.getTime());
  }

  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  return 0;
}
