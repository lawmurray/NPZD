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
#include "bi/method/DistributedMCMC.hpp"
#include "bi/method/ParticleMCMC.hpp"
#include "bi/method/AuxiliaryParticleFilter.hpp"
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MultinomialResampler.hpp"
#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/pdf/AdditiveExpGaussianPdf.hpp"
#include "bi/pdf/ExpGaussianMixturePdf.hpp"

#ifdef USE_CPU
#include "bi/method/StratifiedResampler.inl"
#include "bi/method/MultinomialResampler.inl"
#include "bi/method/Resampler.inl"
#endif

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);
  mpi::communicator world;
  const int rank = world.rank();
  const int size = world.size();

  /* handle command line arguments */
  real T, H, MIN_ESS;
  double SCALE, TEMP, MIN_TEMP, MAX_TEMP, ALPHA, SD;
  int P, INIT_NS, FORCE_NS, OBS_NS, C, A, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, FILTER, RESAMPLER;

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    ("type", po::value(&FILTER)->default_value("ukf"),
        "type of filter to use, 'ukf' or 'pf'")
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
    (",L", po::value(&L)->default_value(0),
        "lookahead for auxiliary particle filter")
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

  /* init stuff */
  #ifndef USE_CPU
  int dev = chooseDevice(rank);
  std::cerr << "Rank " << rank << ": using device " << dev << std::endl;
  #endif
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-3, 1.0e-3);
  NcError ncErr(NcError::silent_nonfatal);

  /* can cause "invalid device function" error if not correct mangled name */
  //cudaFuncSetCacheConfig("_ZN2bi10kernelRK43I9NPZDModelILj1ELj1ELj1EEEEvdd", cudaFuncCachePreferL1);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;
  const int NP = m.getNetSize(P_NODE);

  /* local proposals */
  host_matrix<> Sigma(m.getPrior(P_NODE).cov());
  std::set<int> logs(m.getPrior(P_NODE).getLogs());

  scal(SCALE, diagonal(Sigma));
  /* special case for PZ model */
//  diagonal(Sigma)(0) = pow(0.01,2);
//  diagonal(Sigma)(1) = pow(0.005,2);

  AdditiveExpGaussianPdf<> q(Sigma, logs);
  /* special case for PZ model */
//  AdditiveExpGaussianPdf<> q(Sigma);
  ExpGaussianMixturePdf<> r(NP, logs);
  r.add(m.getPrior(P_NODE));

  /* state */
  if (FILTER.compare("ukf") == 0) {
    P = calcUnscentedKalmanFilterStateSize(m);
  }
  State s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, InputBuffer::O_NODES, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, InputBuffer::P_NODES, INIT_FILE, INIT_NS);

  /* outputs */
  std::stringstream file;
  const int Y = inObs.countUniqueTimes(T);

  file.str("");
  file << OUTPUT_FILE << '.' << rank;
  ParticleMCMCNetCDFBuffer out(m, C, Y, file.str(), NetCDFBuffer::REPLACE);

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

  /* initialise chain */
  inInit.read(s);
  m.getPrior(P_NODE).samples(rng, s.pHostState); // initialise chain
  /* special case for PZ model */
//  s.pHostState(0,0) = 0.2;
//  s.pHostState(0,1) = 0.15;

  s.upload(P_NODE);

  /* set up resampler, filter and MCMC */
  if (FILTER.compare("ukf") == 0) {
    file.str("");
    file << FILTER_FILE << '.' << rank;
    UnscentedKalmanFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);

    BOOST_AUTO(filter, createUnscentedKalmanFilter(m, s, &inForce, &inObs, &tmp));
    BOOST_AUTO(mcmc, createParticleMCMC(m, q, s, rng, filter, T, &out));
    //BOOST_AUTO(dmcmc, createDistributedMCMC(m, r, rng, mcmc));

    mcmc->sample(C, lambda, SD, A);
    //dmcmc->sample(C, lambda);

    /* output diagnostics */
    std::cout << "Rank " << rank << ": " << mcmc->getNumAccepted() << " of " <<
        mcmc->getNumSteps() << " proposals accepted" << std::endl;
    //std::cout << "Rank " << rank << ": " << dmcmc->getNumRemoteAccepted() <<
    //    " of " << dmcmc->getNumRemote() << " remote accepted" << std::endl;
    //std::cout << "Rank " << rank << ": " << dmcmc->getSent() <<
    //    " non-local sent" << std::endl;

    delete mcmc;
    //delete dmcmc;
    delete filter;
  } else {
    file.str("");
    file << FILTER_FILE << '.' << rank;
    ParticleFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);

    StratifiedResampler resam(s, rng);
    BOOST_AUTO(filter, createAuxiliaryParticleFilter(m, s, rng, L, &resam, &inForce, &inObs, &tmp));
    BOOST_AUTO(mcmc, createParticleMCMC(m, q, s, rng, filter, T, &out));
    //BOOST_AUTO(dmcmc, createDistributedMCMC(m, r, rng, mcmc));

    mcmc->sample(C, lambda, SD, A);
    //dmcmc->sample(C, lambda);

    /* output diagnostics */
    std::cout << "Rank " << rank << ": " << mcmc->getNumAccepted() << " of " <<
        mcmc->getNumSteps() << " proposals accepted" << std::endl;
    //std::cout << "Rank " << rank << ": " << dmcmc->getNumRemoteAccepted() <<
    //    " of " << dmcmc->getNumRemote() << " remote accepted" << std::endl;
    //std::cout << "Rank " << rank << ": " << dmcmc->getSent() <<
    //    " non-local sent" << std::endl;

    delete mcmc;
    //delete dmcmc;
    delete filter;
  }

  return 0;
}
