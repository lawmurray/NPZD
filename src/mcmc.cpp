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
#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/ParallelParticleMCMC.hpp"
#include "bi/method/ParticleFilter.hpp"
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"
#include "bi/updater/FUpdater.hpp"
#include "bi/updater/OYUpdater.hpp"
#include "bi/io/MCMCNetCDFWriter.hpp"
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

  /* handle command line arguments */
  real T, H, MIN_ESS;
  double SCALE, TEMP, MIN_TEMP, MAX_TEMP, ALPHA, SD;
  unsigned P, INIT_NS, FORCE_NS, OBS_NS, B, I, C, A, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE, PROPOSAL_FILE,
      RESAMPLER, FILTER;

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    (",B", po::value(&B)->default_value(0), "no. burn steps")
    (",I", po::value(&I)->default_value(1), "interval for drawing samples")
    (",C", po::value(&C), "no. samples to draw")
    (",A", po::value(&A)->default_value(100),
        "no. samples to drawn before adapting proposal")
    ("filter", po::value(&FILTER),
        "filter to use, 'pf' or 'ukf'")
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
  #ifdef NDEBUG
  NcError ncErr(NcError::silent_nonfatal);
  #else
  NcError ncErr(NcError::verbose_nonfatal);
  #endif

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(REAL(H));
  ode_set_rtoler(REAL(1.0e-5));
  ode_set_atoler(REAL(1.0e-5));
  ode_set_nsteps(REAL(200));

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel m;

  /* prior... */
  NPZDPrior x0;

  /* proposal */
  symmetric_matrix Sigma(m.getPSize());
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
  for (i = 0; i < m.getPSize(); ++i) {
    if (i != m.Dsi.id) { // all are log-normal besides deltaS
      logs.insert(i);
    }
  }
  AdditiveExpGaussianPdf<> q(Sigma, logs);

  /* starting distro */
  symmetric_matrix SigmaS(m.getPSize());
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
  vector mu(m.getPSize());
  vector sumMu(m.getPSize());
  symmetric_matrix sumSigma(m.getPSize());
  sumMu.clear();
  sumSigma.clear();

  /* state */
  if (FILTER.compare("ukf") == 0) {
    P = 2*(m.getDSize() + m.getCSize() + m.getRSize()) + 1;
  }
  State s(m, P);

  /* initialise from file... */
  ForwardNetCDFReader<true,true,true,false,true,true,true> in(m, INIT_FILE, INIT_NS);
  in.read(s);

  /* forcings, observations */
  FUpdater fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OYUpdater oyUpdater(m, OBS_FILE, s, OBS_NS);

  /* output */
  std::stringstream file;
  file << OUTPUT_FILE << '.' << rank;
  MCMCNetCDFWriter out(m, file.str().c_str(), C);
  out.sync();

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

  /* report... */
  std::cerr << "Rank " << rank << ": using device " << dev << ", temperature "
      << lambda << std::endl;

  Filter* filter;
  const unsigned D = m.getDSize();
  const unsigned N = D + m.getCSize();
  real mu0[N];
  real Sigma0[N*N];

  if (FILTER.compare("ukf") == 0) {
    /* unscented Kalman filter */
    for (i = 0; i < N*N; ++i) {
      Sigma0[i] = REAL(0.0);
    }
    for (i = 0; i < D; ++i) {
      mu0[i] = x0.getDPrior().mean()(i);
      Sigma0[i*N+i] = x0.getDPrior().cov()(i,i);
    }
    for (i = D; i < N; ++i) {
      mu0[i] = x0.getCPrior().mean()(i - D);
      Sigma0[i*N+i] = x0.getCPrior().cov()(i - D, i - D);
    }
    filter = new UnscentedKalmanFilter<NPZDModel>(m, mu0, Sigma0, s,
        &fUpdater, &oyUpdater);
  } else {
    /* particle filter */
    Resampler* resam;
    if (RESAMPLER.compare("stratified") == 0) {
      resam = new StratifiedResampler(s, rng);
    } else {
      resam = new MetropolisResampler(s, rng, L);
    }
    filter = new ParticleFilter<NPZDModel>(m, s, rng, resam, &fUpdater,
        &oyUpdater);
  }

  /* MCMC */
  real l1, l2;
  State s2(m, 1);
  state_vector theta(s2.pState);
  vector x(m.getPSize());
  bool accepted;

  x0.getPPrior().sample(rng, s.pState); // initialise chain
  //s0.sample(rng, s.pState);
  s.upload();

  ParallelParticleMCMC<NPZDModel,NPZDPrior,AdditiveExpGaussianPdf<> > mcmc(m,
      x0, q, ALPHA, s, rng, filter);

  for (i = 0; i < B+I*C; ++i) {
    l1 = mcmc.getLogLikelihood();
    accepted = mcmc.step(T, lambda);
    theta = mcmc.getState();
    l2 = mcmc.getLastProposedLogLikelihood();

    if (i >= B && (i - B) % I == 0) {
      out.write(s2, l1);
      out.sync();
    }

    std::cerr << rank << '.' << i << ": " << l1 << " -> " << l2;
    if (accepted) {
      std::cerr << " ***accepted***";
    }
    std::cerr << std::endl;

    /* adapt proposal */
    noalias(x) = theta;
    q.log(x);
    noalias(sumMu) += x;
    noalias(sumSigma) += ublas::outer_prod(x,x);

    if (i > A) {
      double sd = SD;
      if (sd <= 0.0) {
        sd = std::pow(2.4,2) / m.getPSize();
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
