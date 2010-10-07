/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "device.hpp"
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/pdf/AdditiveExpGaussianPdf.hpp"
#include "bi/pdf/ExpGaussianMixturePdf.hpp"

#include "bi/method/DistributedMCMC.hpp"
#include "bi/method/ParticleMCMC.hpp"
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/method/AuxiliaryParticleFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MultinomialResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"

#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"

#ifdef USE_CPU
#include "bi/method/StratifiedResampler.inl"
#include "bi/method/MultinomialResampler.inl"
#include "bi/method/MetropolisResampler.inl"
#include "bi/method/Resampler.inl"
#endif

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <sys/time.h>

/**
 * @def SAMPLE
 *
 * Macro to call appropriate sampling function.
 */
#define SAMPLE \
   if (REMOTE) { \
     distributedSample(m, q, r, s, rng, filter, T, &out, C, R, ALPHA, BETA, lambda, SD, A); \
   } else { \
     sample(m, q, s, rng, filter, T, &out, C, lambda, SD, A); \
   }

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

/**
 * Sample using particular filter and resampler.
 */
template<class B, class Q1, class F, class IO1>
void sample(B& m, Q1& q, State& s, Random& rng, F* filter, const real T,
    IO1* out, const int C, const real lambda, const real SD, const int A);

/**
 * Distributed sample using particular filter, resampler and remote proposal.
 */
template<class B, class Q1, class Q2, class F, class IO1>
void distributedSample(B& m, Q1& q, Q2& r, State& s, Random& rng, F* filter, const real T,
    IO1* out, const int C, const int R, const real alpha, const real beta,
    const real lambda, const real SD, const int A);

int main(int argc, char* argv[]) {
  int j;

  /* mpi */
  mpi::environment env(argc, argv);
  mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();

  /* handle command line arguments */
  real T, H, MIN_ESS, EPS_REL, EPS_ABS;
  double SCALE, TEMP, MIN_TEMP, MAX_TEMP, ALPHA, BETA, SD;
  int P, INIT_NS, FORCE_NS, OBS_NS, C, A, R, L, J, SEED;
  bool REMOTE;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, FILTER, RESAMPLER;

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    (",C", po::value(&C), "no. samples to draw")
    (",A", po::value(&A)->default_value(100),
        "no. samples to draw before adapting proposal")
    ("temp", po::value(&TEMP),
        "temperature of chain, if min-temp and max-temp not given")
    ("min-temp", po::value(&MIN_TEMP)->default_value(1.0),
        "minimum temperature in parallel tempering pool")
    ("max-temp", po::value(&MAX_TEMP),
        "maximum temperature in parallel tempering pool");

  po::options_description dmcmcOptions("Distributed MCMC options");
  dmcmcOptions.add_options()
    ("remote", po::value(&REMOTE)->default_value(false),
        "use remote proposal")
    (",R", po::value(&R)->default_value(100),
        "no. samples to draw before incorporating remote proposal")
    ("alpha", po::value(&ALPHA)->default_value(0.1),
        "remote proposal mixing proportion")
    ("beta", po::value(&BETA)->default_value(0.2),
          "remote proposal update propensity");

  po::options_description proposalOptions("Proposal options");
  proposalOptions.add_options()
    ("proposal-file", po::value(&PROPOSAL_FILE),
        "input file, of same format as output of ukf. Covariance at final "
        "time is used as the base proposal covariance, scaled by sd, and "
        "mean is used to initialise the chain")
    ("sd", po::value(&SD)->default_value(0.0),
        "s_d parameter for scaling of covariance of proposal taken from "
        "proposal-file, or adapted if adaptation enabled. Defaults to "
        "2.4^2/N, where N is number of parameters")
    ("scale", po::value(&SCALE)->default_value(0.01),
        "if proposal-file not given, prior is scaled by this value to "
        "obtain the initial proposal");

  po::options_description filterOptions("Filter options");
  filterOptions.add_options()
    (",T", po::value(&T), "total time to filter")
    ("type", po::value(&FILTER)->default_value("ukf"),
        "type of filter to use, 'ukf' or 'pf'");

  po::options_description pfOptions("Particle filter options");
  pfOptions.add_options()
    (",P", po::value(&P), "number of particles")
    ("resampler", po::value(&RESAMPLER)->default_value("stratified"),
        "resampling strategy, 'stratified', 'multinomial' or 'metropolis'")
    (",L", po::value(&L)->default_value(0),
        "number of observations to look ahead (auxiliary particle filter")
    (",J", po::value(&J)->default_value(10),
        "number of steps for Metropolis resampler")
    ("min-ess", po::value(&MIN_ESS)->default_value(1.0),
        "minimum ESS (as proportion of P) at each step to avoid resampling");

  po::options_description odeOptions("ODE numerical integrator options");
  odeOptions.add_options()
    (",h", po::value(&H)->default_value(1.0),
        "suggested first step size")
    ("eps-rel", po::value(&EPS_REL)->default_value(1.0e-3),
        "relative error bound")
    ("eps-abs", po::value(&EPS_ABS)->default_value(1.0e-6),
        "absolute error bound");

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
  desc.add(mcmcOptions).add(dmcmcOptions).add(proposalOptions).add(filterOptions);
  desc.add(pfOptions).add(odeOptions).add(ioOptions);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    if (rank == 0) {
      std::cerr << desc << std::endl;
    }
    return 0;
  }

  /* init stuff */
  int dev = chooseDevice(rank);
  std::cerr << "Rank " << rank << ": using device " << dev << std::endl;
  bi_omp_init();
  bi_ode_init(H, EPS_ABS, EPS_REL);
  h_ode_set_nsteps(100);
  NcError ncErr(NcError::silent_nonfatal);

  /* can cause "invalid device function" error if not correct mangled name */
  //cudaFuncSetCacheConfig("_ZN2bi10kernelRK43I9NPZDModelILj1ELj1ELj1EEEEvdd", cudaFuncCachePreferL1);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;
  const int NP = m.getNetSize(P_NODE);
  const int ND = m.getNetSize(D_NODE);
  const int NC = m.getNetSize(C_NODE);
  if (SD <= 0.0) {
    SD = 2.4*2.4/NP;
  }

  /* state */
  State s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, InputBuffer::O_NODES, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, InputBuffer::P_NODES, INIT_FILE, INIT_NS);
  const int Y = inObs.countUniqueTimes(T) + 1;

  /* outputs */
  std::stringstream file;
  file << OUTPUT_FILE << '.' << rank;
  ParticleMCMCNetCDFBuffer out(m, C, Y, file.str(), NetCDFBuffer::REPLACE);

  /* local proposal and Markov chain initialisation */
  AdditiveExpGaussianPdf<> q(NP);
  if (PROPOSAL_FILE.compare("") != 0) {
    /* construct from input file */
    /**
     * @todo We don't have the correct model specification for the online UKF
     * run here, this will only work if ND + NC >= NP.
     */
    UnscentedKalmanFilterNetCDFBuffer inProposal(m, PROPOSAL_FILE);
    host_vector<real> mu(ND + NC);
    host_matrix<real> Sigma(ND + NC, ND + NC);
    host_matrix<real> transSigma(ND + NC, ND + NC);

    inProposal.readStateMarginal(inProposal.size2() - 1, mu, transSigma);
    transpose(transSigma, Sigma);
    for (j = 0; j < Sigma.size2(); ++j) {
      scal(SD, column(Sigma, j));
    }
    q.setCov(subrange(Sigma, 0, NP, 0, NP));

    /* initialise chain from mean */
    expVec(mu, m.getPrior(P_NODE).getLogs());
    row(s.pHostState, 0) = subrange(mu, 0, NP);
  } else {
    /* construct from scaled prior */
    host_matrix<real> Sigma(NP,NP);
    Sigma = m.getPrior(P_NODE).cov();

    for (j = 0; j < Sigma.size2(); ++j) {
      scal(SCALE, column(Sigma, j));
    }
    //////// add next two lines for PZ model ////////
//    diagonal(Sigma)(0) = pow(0.01,2);
//    diagonal(Sigma)(1) = pow(0.005,2);

    q.setCov(Sigma);

    /* initialise chain from file... */
    //inInit.read(s);

    /* ...or prior... */
    //m.getPrior(P_NODE).samples(rng, s.pHostState);

    /* ...or special case */
    //////// add next two lines for PZ model ////////
//    s.pHostState(0,0) = 0.2;
//    s.pHostState(0,1) = 0.15;
  }
  s.upload(P_NODE);

  ///////// remove next line for PZ model ////////
  q.setLogs(m.getPrior(P_NODE).getLogs());

  /* remote proposal */
  ExpGaussianMixturePdf<> r(NP, m.getPrior(P_NODE).getLogs());
  r.add(m.getPrior(P_NODE));

  /* temperature of chain */
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

  file.str("");
  file << FILTER_FILE << '.' << rank;

  if (FILTER.compare("ukf") == 0) {
    UnscentedKalmanFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);
    BOOST_AUTO(filter, createUnscentedKalmanFilter(m, s, &inForce, &inObs, &tmp));
    SAMPLE
    delete filter;
  } else {
    ParticleFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);
    if (RESAMPLER.compare("stratified") == 0) {
      StratifiedResampler resam(s, rng);
      BOOST_AUTO(filter, createAuxiliaryParticleFilter(m, s, rng, L, &resam, &inForce, &inObs, &tmp));
      SAMPLE
      delete filter;
    } else if (RESAMPLER.compare("multinomial") == 0) {
      MultinomialResampler resam(s, rng);
      BOOST_AUTO(filter, createAuxiliaryParticleFilter(m, s, rng, L, &resam, &inForce, &inObs, &tmp));
      SAMPLE
      delete filter;
    } else if (RESAMPLER.compare("metropolis") == 0) {
      MetropolisResampler resam(s, rng, J);
      BOOST_AUTO(filter, createAuxiliaryParticleFilter(m, s, rng, L, &resam, &inForce, &inObs, &tmp));
      SAMPLE
      delete filter;
    }
  }

  return 0;
}

template<class B, class Q1, class F, class IO1>
void sample(B& m, Q1& q, State& s, Random& rng, F* filter, const real T,
    IO1* out, const int C, const real lambda, const real SD, const int A) {
  mpi::communicator world;
  const int rank = world.rank();

  BOOST_AUTO(mcmc, createParticleMCMC(m, q, s, rng, filter, T, out));

  mcmc->sample(C, lambda, SD, A);

  std::cout << "Rank " << rank << ": " << mcmc->getNumAccepted() << " of " <<
      mcmc->getNumSteps() << " proposals accepted" << std::endl;

  delete mcmc;
}

template<class B, class Q1, class Q2, class F, class IO1>
void distributedSample(B& m, Q1& q, Q2& r, State& s, Random& rng, F* filter, const real T,
    IO1* out, const int C, const int R, const real alpha, const real beta,
    const real lambda, const real SD, const int A) {
  mpi::communicator world;
  const int rank = world.rank();

  BOOST_AUTO(mcmc, createParticleMCMC(m, q, s, rng, filter, T, out));
  BOOST_AUTO(dmcmc, createDistributedMCMC(m, r, rng, mcmc));

  dmcmc->sample(C, R, alpha, beta, lambda);

  std::cout << "Rank " << rank << ": " << mcmc->getNumAccepted() << " of " <<
      mcmc->getNumSteps() << " proposals accepted" << std::endl;
  std::cout << "Rank " << rank << ": " << dmcmc->getNumRemoteAccepted() <<
      " of " << dmcmc->getNumRemote() << " remote accepted" << std::endl;

  delete dmcmc;
  delete mcmc;
}
