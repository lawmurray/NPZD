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
  bool OUTPUT, TIME;

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
    ("output", po::value(&OUTPUT)->default_value(false), "enable output")
    ("time", po::value(&TIME)->default_value(false), "enable timing output");

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
  chooseDevice(rank);

  /* NetCDF error reporting */
  #ifdef NDEBUG
  NcError ncErr(NcError::silent_nonfatal);
  #else
  NcError ncErr(NcError::verbose_nonfatal);
  #endif

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* state */
  State s(m, P);

  /* priors */
  BOOST_AUTO(p0, buildPPrior(m));
  BOOST_AUTO(s0, buildSPrior(m));
  BOOST_AUTO(d0, buildDPrior(m));
  BOOST_AUTO(c0, buildCPrior(m));

  /* proposal */
  BOOST_AUTO(q, buildPProposal(m, SCALE));

  /* forcings, observations */
  FUpdater<> fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OUpdater<> oUpdater(m, OBS_FILE, s, OBS_NS);

  /* output */
  NetCDFWriter<real_t,false,false,false,false,false,false,true>* out;
  if (OUTPUT) {
    out = new NetCDFWriter<real_t,false,false,false,false,false,false,true>(m, OUTPUT_FILE, P, 1, L);
  } else {
    out = NULL;
  }

  /* initialise particle filter */
  init(H, m, s, rng, &fUpdater, &oUpdater);

  /* MCMC */
  double l1, l2, lr, alpha, num, den;
  unsigned i;
  vector theta1(m.getPSize());
  state_vector theta2(s.pState);

  timeval start, end;
  gettimeofday(&start, NULL);

  p0.sample(rng, s.pState);
  for (i = 0; i < B+I*L; ++i) {
    //rng.seed(SEED); // ensures prior samples same between runs
    s0.sample(rng, s.sState);
    d0.sample(rng, s.dState);
    c0.sample(rng, s.cState);

    if (out != NULL && i >= B && (i - B) % I == 0) {
      out->reset();
      out->write(s, 0.0, (i - B) / I);
    }

    /* calculate likelihood */
    l2 = filter(T, MIN_ESS*P);

    std::cerr << i << ": log(L) = " << l2 << ' ';

    if (i == 0) {
      /* first proposal, accept */
      theta1 = theta2;
      l1 = l2;
      std::cerr << "***accept***";
    } else {
      /* accept or reject */
      alpha = rng.uniform(0.0,1.0);
      lr = pow(exp(l2 - l1), pow(TEMP,-1));
      num = pow(p0(theta2), pow(TEMP,-1))*q(theta2,theta1);
      den = pow(p0(theta1), pow(TEMP,-1))*q(theta1,theta2);
      if (alpha < lr*num/den) {
        /* accept */
        theta1 = theta2;
        l1 = l2;
        std::cerr << "***accept***";
      }
    }
    std::cerr << std::endl;

    /* output */
    if (out != NULL && i >= B && (i - B) % I == 0) {
      theta2 = theta1;
      out->write(s, 0.0, (i - B) / I);
    }

    /* next parameter configuration */
    q.sample(rng, theta1, theta2);
  }

  /* final timing results */
  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  /* clean up */
  destroy();
  delete out;

  return 0;
}
