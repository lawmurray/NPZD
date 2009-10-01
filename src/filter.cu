/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "filter.cuh"

#include "bi/cuda/ode/IntegratorConstants.cuh"
#include "bi/method/ParticleFilter.cuh"

using namespace bi;

void filter(const real_t T, NPZDModel& m, State& s, Random& rng,
    Result<>* r, FUpdater<>* fUpdater, OUpdater<>* oUpdater,
    NetCDFWriter<>* out) {
  /* parameters for ODE integrator on GPU */
  ode_init();
  ode_set_h0(CUDA_REAL(0.2));
  ode_set_rtoler(CUDA_REAL(1.0e-3));
  ode_set_atoler(CUDA_REAL(1.0e-3));
  ode_set_nsteps(CUDA_REAL(1000));

  /* particle filter */
  ParticleFilter<NPZDModel,real_t> pf(m, s, rng, r, fUpdater, oUpdater);

  /* output initial state */
  if (out != NULL) {
    out->write(s, pf.getTime());
  }

  /* stream */
  cudaStream_t stream;
  CUDA_CHECKED_CALL(cudaStreamCreate(&stream));

  /* filter */
  pf.bind(stream);
  pf.upload(stream);
  while (pf.getTime() < T) {
    std::cerr << pf.getTime() << ' ';
    pf.advance(T, stream);
    pf.weight(stream);
    pf.resample(stream);
    if (out != NULL) {
      pf.download(stream);
      cudaStreamSynchronize(stream);
      out->write(s, pf.getTime());
    }
  }
  std::cerr << std::endl;

  cudaStreamSynchronize(stream);
  CUDA_CHECKED_CALL(cudaStreamDestroy(stream));
}
