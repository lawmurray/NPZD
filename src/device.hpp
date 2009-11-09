#ifndef DEVICE_HPP
#define DEVICE_HPP

/**
 * Choose CUDA device to use.
 *
 * @param Rank of process.
 *
 * @return Id of device.
 */
int chooseDevice(const int rank);

#endif
