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
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/buffer/SparseInputNetCDFBuffer.hpp"
#include "bi/buffer/UnscentedKalmanFilterNetCDFBuffer.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* openmp */
  bi_omp_init();
  bi_ode_init(1.0, 1.0e-6, 1.0e-3);
  h_ode_set_nsteps(100);

  /* handle command line arguments */
  real T, H;
  unsigned INIT_NS, FORCE_NS, OBS_NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE, RESAMPLER;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",T", po::value(&T), "simulation time for each trajectory")
    (",h", po::value(&H)->default_value(0.2),
        "suggested first step size for each trajectory")
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
  NcError ncErr(NcError::silent_nonfatal);

  /* model */
  NPZDModel<> m;

  /* state and intermediate results */
  int P = 1;
  State s(m, P);

  /* random number generator */
  Random rng(SEED);

  /* inputs */
  SparseInputNetCDFBuffer inForce(m, InputBuffer::F_NODES, FORCE_FILE, FORCE_NS);
  SparseInputNetCDFBuffer inObs(m, InputBuffer::O_NODES, OBS_FILE, OBS_NS);
  SparseInputNetCDFBuffer inInit(m, InputBuffer::P_NODES, INIT_FILE, INIT_NS);

  /* initialise state */
  inInit.read(s);
  s.upload();

  /* output */
  UnscentedKalmanFilterNetCDFBuffer* out;
  if (OUTPUT) {
    out = new UnscentedKalmanFilterNetCDFBuffer(m, P, inObs.countUniqueTimes(T),
        OUTPUT_FILE, NetCDFBuffer::REPLACE);
  } else {
    out = NULL;
  }

  /* set filter */
  BOOST_AUTO(filter, createUnscentedKalmanFilter(m, s, rng, &inForce, &inObs, out));

  /* do filter */
  timeval start, end;
  gettimeofday(&start, NULL);
  filter->filter(T);
  gettimeofday(&end, NULL);

  /* output timing results */
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  delete filter;

  return 0;
}
