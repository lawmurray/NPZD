#include "device.hpp"

#include "bi/cuda/cuda.hpp"

#include <vector>

int chooseDevice(const int rank) {
  int dev, num;
  cudaDeviceProp prop;
  std::vector<int> valid;

  /* build list of valid devices */
  CUDA_CHECKED_CALL(cudaGetDeviceCount(&num));
  for (dev = 0; dev < num; ++dev) {
    CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, dev));
    if (prop.major >= 1 && prop.minor >= 3) { // require compute 1.3 or later
      valid.push_back(dev);
    }
  }
  BI_ERROR(valid.size() > 0, "No devices of at least compute 1.3 available");

  /* select device */
  CUDA_CHECKED_CALL(cudaSetDevice(valid[rank % valid.size()]));
  CUDA_CHECKED_CALL(cudaGetDevice(&dev));

  return dev;
}
