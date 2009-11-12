/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "mcmc.hpp"
#include "prior.hpp"
#include "device.hpp"

#include "bi/method/Simulator.hpp"
#include "bi/math/vector.hpp"
#include "bi/math/matrix.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  const unsigned rank = world.rank();
  const unsigned size = world.size();

  /* handle command line arguments */
  real_t T, H, MIN_ESS;
  double SCALE, TEMP, MIN_TEMP, MAX_TEMP, ALPHA, BETA;
  unsigned P, INIT_NS, FORCE_NS, OBS_NS, B, I, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE, PROPOSAL_FILE;
  bool OUTPUT;

  po::options_description pfOptions("Particle filter options");
  pfOptions.add_options()
    (",P", po::value(&P), "no. particles")
    (",T", po::value(&T), "total time to filter")
    (",h", po::value(&H),
        "suggested first step size for numerical integration")
    ("min-ess", po::value(&MIN_ESS)->default_value(1.0),
        "minimum ESS (as proportion of P) at each step to avoid resampling");

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    (",B", po::value(&B)->default_value(0), "no. burn steps")
    (",I", po::value(&I)->default_value(1), "interval for drawing samples")
    (",L", po::value(&L), "no. samples to draw")
    ("prior-scale", po::value(&SCALE),
        "scale of proposal relative to prior")
    ("temp", po::value(&TEMP)->default_value(1.0),
        "temperature of chain, if min-temp and max-temp not given")
    ("min-temp", po::value(&MIN_TEMP)->default_value(1.0),
        "minimum temperature in parallel tempering pool")
    ("max-temp", po::value(&MAX_TEMP),
        "maximum temperature in parallel tempering pool")
    ("alpha", po::value(&ALPHA)->default_value(0.05),
        "probability of non-local live proposal at each step")
    ("beta", po::value(&BETA)->default_value(0.05),
        "probability of non-local file proposal at each step");

  po::options_description ioOptions("I/O options");
  ioOptions.add_options()
    ("init-file", po::value(&INIT_FILE),
        "input file containing initial values")
    ("force-file", po::value(&FORCE_FILE),
        "input file containing forcings")
    ("obs-file", po::value(&OBS_FILE),
        "input file containing observations")
    ("proposal-file", po::value(&PROPOSAL_FILE),
        "input file containing non-local file proposals")
    ("output-file", po::value(&OUTPUT_FILE),
        "output file to contain results")
    ("init-ns", po::value(&INIT_NS)->default_value(0),
        "index along ns dimension of initial value file to use")
    ("force-ns", po::value(&FORCE_NS)->default_value(0),
        "index along ns dimension of forcings file to use")
    ("obs-ns", po::value(&OBS_NS)->default_value(0),
        "index along ns dimension of observations file to use")
    ("output", po::value(&OUTPUT)->default_value(false), "enable output");

  po::options_description desc("General options");
  desc.add_options()
      ("help", "produce help message")
      ("seed", po::value(&SEED)->default_value(0),
          "pseudorandom number seed");
  desc.add(pfOptions).add(mcmcOptions).add(ioOptions);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    if (rank == 0) {
      std::cerr << desc << std::endl;
    }
    return 0;
  }

  /* select CUDA device */
  int dev = chooseDevice(rank);
  std::cerr << "Rank " << rank << " using device " << dev << std::endl;

  /* NetCDF error reporting */
  #ifdef NDEBUG
  NcError ncErr(NcError::silent_nonfatal);
  #else
  NcError ncErr(NcError::verbose_nonfatal);
  #endif

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(H));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(CUDA_REAL(200));

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* prior */
  NPZDPrior x0;

  /* proposal */
  BOOST_AUTO(q, buildPProposal(m, SCALE));

  /* temperature */
  double lambda;
  if (vm.count("temp")) {
    lambda = TEMP;
  } else if (vm.count("min-temp") && vm.count("max-temp")) {
    if (size > 1) {
      lambda = MIN_TEMP + rank*(MAX_TEMP - MIN_TEMP) / (size - 1);
    } else {
      lambda = MIN_TEMP;
    }
  } else {
    lambda = 1.0;
  }

  /* state */
  State s(m, P);

  /* forcings, observations */
  FUpdater fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OUpdater oUpdater(m, OBS_FILE, s, OBS_NS);

  /* output */
  MCMCNetCDFWriter* out;
  if (OUTPUT) {
    out = new MCMCNetCDFWriter(m, OUTPUT_FILE, L);
  } else {
    out = NULL;
  }

  /* initialise MCMC */
  init(m, x0, q, s, rng, &fUpdater, &oUpdater);

  /* do MCMC */
  double l;
  State s2(m, 1);
  state_vector theta(s2.pState);
  bool accepted;
  unsigned i;

  for (i = 0; i < B+I*L; ++i) {
    accepted = step(T, MIN_ESS*P, lambda, theta, l);
    if (out != NULL && i >= B && (i - B) % I == 0) {
      out->write(s2, l);
    }

    std::cerr << i << ": " << l;
    if (accepted) {
      std::cerr << " ***accepted***";
    }
    std::cerr << std::endl;
  }

  /* clean up */
  destroy();
  delete out;

  return 0;
}
