/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/ParticleFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
//#include "bi/method/MetropolisResampler.hpp"
#include "bi/updater/FUpdater.hpp"
#include "bi/updater/OYUpdater.hpp"
#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"

#include "cuda_runtime.h"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* openmp */
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-3, 1.0e-3);

  /* handle command line arguments */
  real T, H, MIN_ESS;
  unsigned P, L, INIT_NS, FORCE_NS, OBS_NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE, RESAMPLER;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. particles")
    (",T", po::value(&T), "simulation time for each trajectory")
    (",h", po::value(&H)->default_value(0.2),
        "suggested first step size for each trajectory")
    ("min-ess", po::value(&MIN_ESS)->default_value(1.0),
        "minimum ESS (as proportion of P) at each step to avoid resampling")
    ("seed", po::value(&SEED)->default_value(0),
        "pseudorandom number seed")
    ("resampler", po::value(&RESAMPLER)->default_value("metropolis"),
        "resampling strategy, 'stratified' or 'metropolis'")
    (",L", po::value(&L)->default_value(20),
        "no. steps for Metropolis resampler")
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

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;

  /* prior over initial conditions */
  NPZDPrior prior(m);

  /* state and intermediate results */
  State s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, InputBuffer::O_NODES, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, InputBuffer::P_NODES, INIT_FILE, INIT_NS);

  /* initialise state */
  inInit.read(s);
  prior.getDPrior().sample(rng, s.dHostState);
  prior.getCPrior().sample(rng, s.cHostState);
  //prior.getPPrior().sample(rng, s.pHostState);
  s.upload();

  /* output */
  ParticleFilterNetCDFBuffer* out;
  if (OUTPUT) {
    out = new ParticleFilterNetCDFBuffer(m, P, inObs.numUniqueTimes(),
        OUTPUT_FILE, NetCDFBuffer::REPLACE);
  } else {
    out = NULL;
  }

  /* randoms, forcings, observations */
  FUpdater fUpdater(s, inForce);
  OYUpdater oyUpdater(s, inObs);

  /* resamplers */
//  bool isStratified, isMetropolis;
//  Resampler* resam;
//  if (RESAMPLER.compare("stratified") == 0) {
//    isStratified = true;
//    isMetropolis = false;
    StratifiedResampler resam(s, rng);
//  } else {
//    isStratified = false;
//    isMetropolis = true;
//    //resam = new MetropolisResampler(s, rng, L);
//    BI_ERROR(false, "Metropolis resampler not available at present");
//  }

  /* particle filter */
  ParticleFilter<NPZDModel<>, StratifiedResampler> pf(m, s, rng, &resam, &fUpdater, &oyUpdater, out);

  /* filter */
  timeval start, end;
  gettimeofday(&start, NULL);
  pf.filter(T);
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  return 0;
}
