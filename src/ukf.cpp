/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "prior.hpp"
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/method/UnscentedKalmanFilter.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OYUpdater.hpp"
#include "bi/io/ForwardNetCDFReader.hpp"
#include "bi/io/ForwardNetCDFWriter.hpp"

#include "cuda_runtime.h"

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

  /* handle command line arguments */
  real_t T, H;
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

  /* report missing variables in NetCDF, but don't die */
  #ifndef NDEBUG
  NcError ncErr(NcError::verbose_nonfatal);
  #else
  NcError ncErr(NcError::silent_nonfatal);
  #endif

  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(H));
  ode_set_rtoler(CUDA_REAL(1.0e-5));
  ode_set_atoler(CUDA_REAL(1.0e-5));
  ode_set_nsteps(1000);

  /* model */
  NPZDModel m;

  /* prior */
  NPZDPrior prior;

  /* state and intermediate results */
  unsigned P = 2*(m.getDSize() + m.getCSize() + m.getRSize()) + 1;
  State s(m, P);

  /* initialise from file... */
  ForwardNetCDFReader<true,true,true,false,true,true,true> in(m, INIT_FILE, INIT_NS);
  in.read(s);

  /* forcings, observations */
  FUpdater fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OYUpdater oyUpdater(m, OBS_FILE, s, OBS_NS);

  /* outputs */
  ForwardNetCDFWriter* out;
  if (OUTPUT) {
    out = new ForwardNetCDFWriter(m, OUTPUT_FILE, P, oyUpdater.numUniqueTimes());
  } else {
    out = NULL;
  }

  /* prior */
  const unsigned D = m.getDSize();
  const unsigned C = m.getCSize();
  const unsigned N = D + C;
  unsigned i;
  real_t mu[N];
  real_t Sigma[N*N];
  for (i = 0; i < N*N; ++i) {
    Sigma[i] = CUDA_REAL(0.0);
  }
  for (i = 0; i < D; ++i) {
    mu[i] = prior.getDPrior().mean()(i);
    Sigma[i*N+i] = prior.getDPrior().cov()(i,i);
  }
  for (i = D; i < N; ++i) {
    mu[i] = prior.getCPrior().mean()(i - D);
    Sigma[i*N+i] = prior.getCPrior().cov()(i - D, i - D);
  }

  /* filter */
  timeval start, end;
  gettimeofday(&start, NULL);

  UnscentedKalmanFilter<NPZDModel> ukf(m, mu, Sigma, s, &fUpdater, &oyUpdater);
  cudaThreadSynchronize();

  /* filter */
  cudaStream_t stream;
  CUDA_CHECKED_CALL(cudaStreamCreate(&stream));
  ukf.bind(stream);
  ukf.upload(stream);
  while (ukf.getTime() < T) {
    BI_LOG("t = " << ukf.getTime());
    ukf.predict(T, stream);
    ukf.correct(stream);

    if (out != NULL) {
      ukf.download(stream);
      cudaStreamSynchronize(stream);
      out->write(s, ukf.getTime());
    }
  }
  cudaStreamSynchronize(stream);
  ukf.unbind();
  CUDA_CHECKED_CALL(cudaStreamDestroy(stream));

  /* wrap up timing */
  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  delete out;
  return 0;
}
