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
#include "bi/method/StratifiedResampler.hpp"
#include "bi/method/MultinomialResampler.hpp"
#include "bi/method/MetropolisResampler.hpp"
#include "bi/buffer/ParticleFilterNetCDFBuffer.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/pdf/AdditiveExpGaussianPdf.hpp"
#include "bi/pdf/ExpGaussianMixturePdf.hpp"

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

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);
  mpi::communicator world;
  const int rank = world.rank();

  /* handle command line arguments */
  real T;
  int P, INIT_NS, FORCE_NS, OBS_NS, C, L;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, FILTER_FILE, OUTPUT_FILE,
      PROPOSAL_FILE, RESAMPLER;

  po::options_description mcmcOptions("MCMC options");
  mcmcOptions.add_options()
    (",C", po::value(&C), "no. samples to draw");

  po::options_description pfOptions("Particle filter options");
  pfOptions.add_options()
    (",P", po::value(&P), "no. particles")
    (",T", po::value(&T), "total time to filter")
    ("resampler", po::value(&RESAMPLER)->default_value("metropolis"),
        "resampling strategy, 'stratified' or 'metropolis'")
    (",L", po::value(&L)->default_value(0),
        "lookahead for auxiliary particle filter");

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
  bi_ode_init(1.0, 1.0e-6, 1.0e-3);
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

  /* state */
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

  file.str("");
  file << FILTER_FILE << '.' << rank;
  ParticleFilterNetCDFBuffer tmp(m, P, Y, file.str(), NetCDFBuffer::REPLACE);

  /* set up resampler, filter and MCMC */
  StratifiedResampler resam(s, rng);
  //MultinomialResampler resam(s, rng);
  //MetropolisResampler resam(s, rng, 5);
  BOOST_AUTO(filter, createAuxiliaryParticleFilter(m, s, rng, L, &resam, &inForce, &inObs, &tmp));
  BOOST_AUTO(mcmc, createParticleMCMC(m, m.getPrior(P_NODE), s, rng, filter, T, &out));

  /* and go... */
  inInit.read(s);
  std::cin >> s.pHostState(0,0) >> s.pHostState(0,1);
  //m.getPrior(P_NODE).samples(rng, s.pHostState); // initialise chain
  s.upload(P_NODE);

  /* using high-level interface */
  //mcmc->sample(C);

  /* using low-level interface */
  host_vector<real> theta(m.getNetSize(P_NODE));
  int c;
  mcmc->init();
  mcmc->output(0);
  for (c = 1; c < C; ++c) {
    //mcmc->m.getPrior(P_NODE).sample(rng, theta);
    std::cin >> theta(0) >> theta(1);

    mcmc->propose(theta);
    mcmc->likelihood();
    mcmc->accept();
    mcmc->output(c);

    /* verbose output */
    std::cerr << c << ":\t";
    std::cerr.width(10);
    std::cerr << mcmc->getState().ll;
    std::cerr << '\t';
    std::cerr.width(10);
    std::cerr << mcmc->getState().p;
    std::cerr << std::endl;
  }
  mcmc->term();

  /* output diagnostics */
  std::cout << "Rank " << rank << ": " << mcmc->getNumAccepted() << " of " <<
      mcmc->getNumSteps() << " proposals accepted" << std::endl;

  delete mcmc;
  delete filter;

  return 0;
}
