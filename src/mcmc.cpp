/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "device.hpp"
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/ParallelParticleMCMC.hpp"
#include "bi/method/ParticleFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"
#include "bi/updater/FUpdater.hpp"
#include "bi/updater/OYUpdater.hpp"
#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/pdf/AdditiveExpGaussianPdf.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);
  mpi::communicator world;
  const unsigned rank = world.rank();
  const unsigned size = world.size();

  /* openmp */
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-3, 1.0e-3);

  /* handle command line arguments */
  real T, H, MIN_ESS;
  double SCALE, TEMP, MIN_TEMP, MAX_TEMP, ALPHA, SD;
  unsigned P, INIT_NS, FORCE_NS, OBS_NS, B, I, C, A, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, RESAMPLER;

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    (",C", po::value(&C), "no. samples to draw")
    (",A", po::value(&A)->default_value(100),
        "no. samples to drawn before adapting proposal")
    ("scale", po::value(&SCALE),
        "scale of proposal relative to prior")
    ("temp", po::value(&TEMP),
        "temperature of chain, if min-temp and max-temp not given")
    ("min-temp", po::value(&MIN_TEMP)->default_value(1.0),
        "minimum temperature in parallel tempering pool")
    ("max-temp", po::value(&MAX_TEMP),
        "maximum temperature in parallel tempering pool")
    ("alpha", po::value(&ALPHA)->default_value(0.05),
        "probability of non-local proposal at each step")
    ("sd", po::value(&SD)->default_value(0.0),
        "s_d parameter for proposal adaptation. Defaults to 2.4^2/d");

  po::options_description pfOptions("Particle filter options");
  pfOptions.add_options()
    (",P", po::value(&P), "no. particles")
    (",T", po::value(&T), "total time to filter")
    ("resampler", po::value(&RESAMPLER)->default_value("metropolis"),
        "resampling strategy, 'stratified' or 'metropolis'")
    (",L", po::value(&L)->default_value(15),
        "no. steps for Metropolis resampler")
    (",h", po::value(&H),
        "suggested first step size for numerical integration")
    ("min-ess", po::value(&MIN_ESS)->default_value(1.0),
        "minimum ESS (as proportion of P) at each step to avoid resampling");

  po::options_description ioOptions("I/O options");
  ioOptions.add_options()
    ("init-file", po::value(&INIT_FILE),
        "input file containing initial values")
    ("force-file", po::value(&FORCE_FILE),
        "input file containing forcings")
    ("obs-file", po::value(&OBS_FILE),
        "input file containing observations")
    ("filter-file", po::value(&FILTER_FILE),
        "temporary file for storage of intermediate particle filter results")
    ("proposal-file", po::value(&PROPOSAL_FILE),
        "input file containing non-local file proposals")
    ("output-file", po::value(&OUTPUT_FILE),
        "output file to contain results")
    ("init-ns", po::value(&INIT_NS)->default_value(0),
        "index along ns dimension of initial value file to use")
    ("force-ns", po::value(&FORCE_NS)->default_value(0),
        "index along ns dimension of forcings file to use")
    ("obs-ns", po::value(&OBS_NS)->default_value(0),
        "index along ns dimension of observations file to use");

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
  #ifndef USE_CPU
  int dev = chooseDevice(rank);
  std::cerr << "Rank " << rank << ": using device " << dev << std::endl;
  #endif

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;
  const unsigned NP = m.getNetSize(P_NODE);
  const unsigned ND = m.getNetSize(D_NODE);
  const unsigned NC = m.getNetSize(C_NODE);

  /* prior */
  NPZDPrior prior(m);

  /* proposal */
  symmetric_matrix Sigma(NP);
  Sigma.clear();
  BOOST_AUTO(d, diag(Sigma));

  d(m.getNode(P_NODE, "Kw")->getId()) = 0.2;
  d(m.getNode(P_NODE, "KCh")->getId()) = 0.3;
  d(m.getNode(P_NODE, "Dsi")->getId()) = 1.0;
  d(m.getNode(P_NODE, "ZgD")->getId()) = 0.1;
  d(m.getNode(P_NODE, "PDF")->getId()) = 0.2;
  d(m.getNode(P_NODE, "ZDF")->getId()) = 0.1;
  d(m.getNode(P_NODE, "muPgC")->getId()) = 0.63;
  d(m.getNode(P_NODE, "muPgR")->getId()) = 0.2;
  d(m.getNode(P_NODE, "muPCh")->getId()) = 0.37;
  d(m.getNode(P_NODE, "muPaN")->getId()) = 1.0;
  d(m.getNode(P_NODE, "muPRN")->getId()) = 0.3;
  d(m.getNode(P_NODE, "muZin")->getId()) = 0.7;
  d(m.getNode(P_NODE, "muZCl")->getId()) = 1.3;
  d(m.getNode(P_NODE, "muZgE")->getId()) = 0.25;
  d(m.getNode(P_NODE, "muDre")->getId()) = 0.5;
  d(m.getNode(P_NODE, "muZmQ")->getId()) = 1.0;

  d *= SCALE;
  d = element_prod(d,d); // square to get variances

  unsigned i;
  std::set<unsigned> logs;
  for (i = 0; i < NP; ++i) {
    if (i != m.getNode(P_NODE, "Dsi")->getId()) { // all log-normal besides this
      logs.insert(i);
    }
  }
  AdditiveExpGaussianPdf<> q(Sigma, logs);

  /* proposal adaptation */
  vector mu(NP);
  vector sumMu(NP);
  symmetric_matrix sumSigma(NP);
  sumMu.clear();
  sumSigma.clear();

  /* state */
  State s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, InputBuffer::O_NODES, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, InputBuffer::P_NODES, INIT_FILE, INIT_NS);

  /* outputs */
  std::stringstream file;
  const unsigned Y = inObs.numUniqueTimes();

  file.str("");
  file << OUTPUT_FILE << '.' << rank;
  ParticleMCMCNetCDFBuffer out(m, C, Y, file.str(), NetCDFBuffer::REPLACE);

  file.str("");
  file << FILTER_FILE << '.' << rank;
  ParticleFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);

  /* temperature */
  real lambda;
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
  std::cerr << "Rank " << rank << ": using temperature " << lambda << std::endl;

  /* randoms, forcings, observations */
  FUpdater fUpdater(s, inForce);
  OYUpdater oyUpdater(s, inObs);

  /* set up resampler, filter and MCMC */
  typedef StratifiedResampler ResamplerType;
  //typedef MetropolisResampler ResamplerType;
  typedef ParticleFilter<NPZDModel<>, ResamplerType> FilterType;
  typedef ParallelParticleMCMC<NPZDModel<>,NPZDPrior,AdditiveExpGaussianPdf<>,FilterType> MCMCType;

  StratifiedResampler resam(s, rng);
  //MetropolisResampler resam(s, rng, L);

  FilterType filter(m, s, rng, &resam, &fUpdater, &oyUpdater, &tmp);
  MCMCType mcmc(m, prior, q, ALPHA, s, rng, &filter, &out);

  /* and go... */
  real l1, l2, p1, p2;
  host_vector<> x(NP);
  shallow_vector y(x);
  bool accepted;

  inInit.read(s);
  prior.getPPrior().sample(rng, s.pHostState); // initialise chain

  for (i = 0; i < C; ++i) {
    accepted = mcmc.step(T, lambda);

    l1 = mcmc.getLogLikelihood();
    p1 = mcmc.getPriorDensity();
    l2 = mcmc.getOtherLogLikelihood();
    p2 = mcmc.getOtherPriorDensity();

    std::cerr << rank << '.' << i << ":\t";
    std::cerr.width(10);
    std::cerr << l1;
    std::cerr << "\tbeats\t";
    std::cerr.width(10);
    std::cerr << l2;
    std::cerr << '\t';
    if (accepted) {
      std::cerr << "accept";
    }
    std::cerr << '\t';
    if (mcmc.wasLastNonLocal()) {
      std::cerr << "non-local";
    }
    std::cerr << std::endl;

    /* adapt proposal */
    x = mcmc.getState();
    q.log(y);
    noalias(sumMu) += y;
    noalias(sumSigma) += ublas::outer_prod(y,y);

    if (i > A) {
      double sd = SD;
      if (sd <= 0.0) {
        sd = std::pow(2.4,2) / m.getNetSize(P_NODE);
      }

      noalias(mu) = sumMu / (i + 1.0);
      noalias(Sigma) = sd*((sumSigma - (i + 1.0)*ublas::outer_prod(mu,mu))/i);
      q.setCov(Sigma);
      mcmc.setProposal(q);
    }
  }

  /* output diagnostics */
  std::cout << "Rank " << rank << ": " << mcmc.getNumAccepted() << " of " <<
      mcmc.getNumSteps() << " proposals accepted" << std::endl;
  std::cout << "Rank " << rank << ": " << mcmc.getNumNonLocalAccepted() <<
      " of " << mcmc.getNumNonLocal() << " non-local accepted" << std::endl;
  std::cout << "Rank " << rank << ": " << mcmc.getNumNonLocalSent() <<
      " non-local sent" << std::endl;

  return 0;
}
