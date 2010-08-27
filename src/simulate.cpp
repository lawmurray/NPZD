/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/math/ode.hpp"
#include "bi/random/Random.hpp"
#include "bi/updater/StochasticRUpdater.hpp"
#include "bi/method/Simulator.hpp"
#include "bi/method/Sampler.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/buffer/SimulatorNetCDFBuffer.hpp"

#include "boost/program_options.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#include "boost/numeric/ublas/io.hpp"
#include "boost/typeof/typeof.hpp"

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

void test();

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* bi init */
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-3, 1.0e-3);

  /* command line arguments */
  real T;
  unsigned P, K, NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. trajectories")
    (",K", po::value(&K), "size of intermediate result buffer")
    (",T", po::value(&T), "simulation time for each trajectory")
    ("seed", po::value(&SEED)->default_value(time(NULL)),
        "pseudorandom number seed")
    ("init-file", po::value(&INIT_FILE),
        "input file containing initial values")
    ("force-file", po::value(&FORCE_FILE),
        "input file containing forcings")
    ("output-file", po::value(&OUTPUT_FILE),
        "output file to contain results")
    ("ns", po::value(&NS)->default_value(0),
        "index along ns dimension in input file to use")
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

  /* state */
  State s(m, P);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, NS);
  SparseInputNetCDFBuffer inInit(m,
      InputBuffer::C_NODES|InputBuffer::D_NODES|InputBuffer::P_NODES,
      INIT_FILE, NS);

  /* initialise state */
  inInit.read(s);
  s.upload();

  /* output */
  SimulatorNetCDFBuffer* out;
  if (OUTPUT) {
    out = new SimulatorNetCDFBuffer(m, P, K, OUTPUT_FILE,
        NetCDFBuffer::REPLACE);
  } else {
    out = NULL;
  }

  /* static sampler & dynamic simulator */
  StochasticRUpdater<NPZDModel<> > rUpdater(s, rng);
  Sampler<NPZDModel<> > sam(m, s, &rUpdater);
  BOOST_AUTO(sim, createSimulator(m, s, &rUpdater, &inForce, out));

  /* simulate and output */
  timeval start, end;
  gettimeofday(&start, NULL);
  sam.sample(); // set static variables
  sim->simulate(T); // simulate dynamic variables
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  delete sim;
  return 0;
}
