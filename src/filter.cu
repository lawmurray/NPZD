/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 293 $
 * $Date: 2009-09-21 11:25:09 +0800 (Mon, 21 Sep 2009) $
 */
#include "filter.cuh"

#include "bi/method/ParticleFilter.cuh"

#include <fstream>

using namespace bi;

void filter(const real_t T, const real_t h, NPZDModel& m, State& s,
    Random& rng, FUpdater* fUpdater, OUpdater* oUpdater,
    ForwardNetCDFWriter* out) {
  /* particle filter */
  ParticleFilter<NPZDModel> pf(m, s, rng, fUpdater, oUpdater);

  /* output initial state */
  if (out != NULL) {
    out->write(s, pf.getTime());
  }

  /* stream */
  cudaStream_t stream;
  CUDA_CHECKED_CALL(cudaStreamCreate(&stream));

  /* ESS output */
  std::ofstream essOut("ess.txt");

  /* filter */
  real_t ess;
  pf.bind(stream);
  pf.upload(stream);
  while (pf.getTime() < T) {
    BI_LOG("t = " << pf.getTime());
    pf.advance(T, stream);
    pf.weight(stream);
    ess = pf.ess(stream);
    essOut << ess << std::endl;
    //BI_LOG("ess = " << ess);
    //if (ess < 0.5*s.P) {
    pf.resample(0.5*s.P, stream);
    //pf.resample(stream);
    //}
    if (out != NULL) {
      pf.download(stream);
      cudaStreamSynchronize(stream);
      out->write(s, pf.getTime());
    }
  }
  //std::cerr << std::endl;

  cudaStreamSynchronize(stream);
  pf.unbind();
  CUDA_CHECKED_CALL(cudaStreamDestroy(stream));
}
