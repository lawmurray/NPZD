/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/StochasticRUpdater.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OYUpdater.hpp"
#include "bi/method/Simulator.hpp"
#include "bi/method/Sampler.hpp"
#include "bi/io/ForwardNetCDFReader.hpp"
#include "bi/io/ForwardNetCDFWriter.hpp"

#include "boost/program_options.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* command line arguments */
  real_t T;
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
    ("obs-file", po::value(&OBS_FILE),
        "input file containing observations")
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

  /* random number generator */
  Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  NcError ncErr(NcError::verbose_nonfatal);

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));

  /* model */
  NPZDModel m;

  /* state */
  State s(m, P);
  ForwardNetCDFReader<true,true,true,false,false,false,true> in(m, INIT_FILE, NS);
  in.read(s); // initialise state

  /* intermediate result buffer */
  Result r(m, P, K);

  /* output */
  ForwardNetCDFWriter* out;
  if (OUTPUT) {
    out = new ForwardNetCDFWriter(m, OUTPUT_FILE, P, K + 1);
  }

  /* static sampler & dynamic simulator */
  StochasticRUpdater<NPZDModel> rUpdater(s, rng);
  FUpdater fUpdater(m, FORCE_FILE, s, NS);
  OYUpdater oyUpdater(m, OBS_FILE, s, NS);
  Sampler<NPZDModel> sam(m, s, &rUpdater);
  Simulator<NPZDModel> sim(m, s, &r, &rUpdater, &fUpdater, &oyUpdater);

  /* simulate and output */
  timeval start, end;
  gettimeofday(&start, NULL);
  sam.sample(); // set static variables
  sim.simulate(T); // simulate dynamic variables
  if (OUTPUT) {
    out->write(r, K);
    out->write(s, sim.getTime());
  }
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  return 0;
}
