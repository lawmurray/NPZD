/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "mcmc.hpp"
#include "prior.hpp"

#include "bi/math/vector.hpp"
#include "bi/math/matrix.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"

#include <iostream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;

using namespace bi;

int main(int argc, char* argv[]) {
  /* handle command line arguments */
  real_t T, H, SCALE;
  unsigned P, INIT_NS, FORCE_NS, OBS_NS, B, I, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. particles")
    (",T", po::value(&T), "simulation time for each trajectory")
    (",h", po::value(&H)->default_value(0.2),
        "suggested first step size for each trajectory")
    (",B", po::value(&B)->default_value(0), "no. burn steps")
    (",I", po::value(&I)->default_value(1), "interval for sample drawings")
    (",L", po::value(&L), "no. samples to draw")
    ("scale", po::value(&SCALE), "scale of proposal relative to prior")
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

  /* NetCDF error reporting */
  #ifndef NDEBUG
  NcError ncErr(NcError::verbose_nonfatal);
  #else
  NcError ncErr(NcError::silent_nonfatal);
  #endif

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* state */
  State s(m, P);

  /* priors */
  BOOST_AUTO(p0, buildPPrior(m));
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
    out = new NetCDFWriter<real_t,false,false,false,false,false,false,true>(m,
        OUTPUT_FILE, 1, 1, L);
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
    /* sample initial conditions */
    d0.sample(rng, s.dState);
    c0.sample(rng, s.cState);

    /* calculate likelihood */
    l2 = filter(T, 0.5*P);

    std::cerr << i << ": log(L) = " << l2 << ' ';

    if (i == 0) {
      /* first proposal, accept */
      theta1 = theta2;
      l1 = l2;
      std::cerr << "accept" << std::endl;
    } else {
      /* accept or reject */
      alpha = rng.uniform(0.0,1.0);
      lr = exp(l2 - l1);
      num = p0(theta2)*q(theta2,theta1);
      den = p0(theta1)*q(theta1,theta2);
      if (alpha < lr*num/den) {
        /* accept */
        theta1 = theta2;
        l1 = l2;
        std::cerr << "accept" << std::endl;
      } else {
        std::cerr << "reject" << std::endl;
      }
    }

    /* output */
    if (out != NULL && i >= B && (i - B) % I == 0) {
      theta2 = theta1;
      out->write(s, 0.0, (i - B) / I);
    }

    /* next parameter configuration */
    q.sample(rng, theta1, theta2);
    //p0.sample(rng, theta2);
  }

  /* final timing results */
  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  /* clean up */
  destroy();

  return 0;
}
