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
  int dev = chooseDevice(rank);

  /* NetCDF error reporting */
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;

  /* prior */
  NPZDPrior prior(m);

  /* proposal */
  symmetric_matrix Sigma(m.getNetSize(P_NODE));
  Sigma.clear();
  BOOST_AUTO(d, diag(Sigma));

  d(m.Kw.id) = 0.2;
  d(m.KCh.id) = 0.3;
  d(m.Dsi.id) = 1.0;
  d(m.ZgD.id) = 0.1;
  d(m.PDF.id) = 0.2;
  d(m.ZDF.id) = 0.1;
  d(m.muPgC.id) = 0.63;
  d(m.muPgR.id) = 0.2;
  d(m.muPCh.id) = 0.37;
  d(m.muPaN.id) = 1.0;
  d(m.muPRN.id) = 0.3;
  d(m.muZin.id) = 0.7;
  d(m.muZCl.id) = 1.3;
  d(m.muZgE.id) = 0.25;
  d(m.muDre.id) = 0.5;
  d(m.muZmQ.id) = 1.0;

  d *= SCALE;
  d = element_prod(d,d); // square to get variances

  unsigned i;
  std::set<unsigned> logs;
  for (i = 0; i < m.getNetSize(P_NODE); ++i) {
    if (i != m.Dsi.id) { // all are log-normal besides deltaS
      logs.insert(i);
    }
  }
  AdditiveExpGaussianPdf<> q(Sigma, logs);

  /* starting distro */
  symmetric_matrix SigmaS(m.getNetSize(P_NODE));
  SigmaS.clear();
  BOOST_AUTO(dS, diag(SigmaS));

  dS(m.Kw.id) = 0.2;
  dS(m.KCh.id) = 0.3;
  dS(m.Dsi.id) = 1.0;
  dS(m.ZgD.id) = 0.1;
  dS(m.PDF.id) = 0.2;
  dS(m.ZDF.id) = 0.1;
  dS(m.muPgC.id) = 0.63;
  dS(m.muPgR.id) = 0.2;
  dS(m.muPCh.id) = 0.37;
  dS(m.muPaN.id) = 1.0;
  dS(m.muPRN.id) = 0.3;
  dS(m.muZin.id) = 0.7;
  dS(m.muZCl.id) = 1.3;
  dS(m.muZgE.id) = 0.25;
  dS(m.muDre.id) = 0.5;
  dS(m.muZmQ.id) = 1.0;

  dS *= 0.1;
  dS = element_prod(dS,dS); // square to get variances

  ExpGaussianPdf<> s0(x0.getPPrior().mean(), SigmaS, logs);

  /* proposal adaptation */
  vector mu(m.getNetSize(P_NODE));
  vector sumMu(m.getNetSize(P_NODE));
  symmetric_matrix sumSigma(m.getNetSize(P_NODE));
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

  /* randoms, forcings, observations */
  FUpdater fUpdater(s, inForce);
  OYUpdater oyUpdater(s, inObs);

  /* set up resampler and filter */
  Resampler* resam;
  Filter* filter;
  if (RESAMPLER.compare("stratified") == 0) {
    resam = new StratifiedResampler(s, rng);
    filter = new ParticleFilter<NPZDModel<>, StratifiedResampler>(m, s, rng, (StratifiedResampler*)resam, &fUpdater, &oyUpdater, &tmp);
  } else {
    resam = new MetropolisResampler(s, rng, L);
    filter = new ParticleFilter<NPZDModel<>, MetropolisResampler>(m, s, rng, (MetropolisResampler*)resam, &fUpdater, &oyUpdater, &tmp);
  }

  /* set up MCMC */
  const unsigned D = m.getNetSize(D_NODE);
  const unsigned N = D + m.getNetSize(C_NODE);
  real mu0[N];
  real Sigma0[N*N];
  real l1, l2;
  vector x(m.getNetSize(P_NODE));
  bool accepted;

  inInit.read(s);
  x0.getPPrior().sample(rng, s.pHostState); // initialise chain
  //s0.sample(rng, s.pState);
  s.upload();

  ParallelParticleMCMC<NPZDModel<>,NPZDPrior,AdditiveExpGaussianPdf<> > mcmc(m,
      x0, q, ALPHA, s, rng, filter);

  std::cerr << "Rank " << rank << ": using device " << dev << ", temperature "
      << lambda << std::endl;

  /* and go... */
  for (i = 0; i < C; ++i) {
    accepted = mcmc.step(T, lambda);

    l1 = mcmc.getLogLikelihood();
    l2 = mcmc.getLastProposedLogLikelihood();

    std::cerr << rank << '.' << i << ": " << l1 << " -> " << l2;
    if (accepted) {
      std::cerr << " ***accepted***";
    }
    std::cerr << std::endl;

    /* adapt proposal */
    noalias(x) = mcmc.getState();
    q.log(x);
    noalias(sumMu) += x;
    noalias(sumSigma) += ublas::outer_prod(x,x);

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
  std::cout << "Rank " << rank << ": " << mcmc.getNumNonLocalAccepts() <<
      " of " << mcmc.getNumNonLocal() << " non-local accepted" << std::endl;
  std::cout << "Rank " << rank << ": " << mcmc.getNumNonLocalResponses() <<
      " of " << mcmc.getNumNonLocalRequests() << " outgoing furnished" <<
      std::endl;
  std::cout << "Rank " << rank << ": " << mcmc.getNumNonLocalFurnishes() <<
      " incoming furnished" << std::endl;

  return 0;
}
