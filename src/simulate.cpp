/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Simulate trajectories from Lorenz96Model.
 */
#include "bi/cuda/cuda.hpp"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;

extern int simulate(const unsigned P, const unsigned K,
    const real_t T, const bool write);

int main(int argc, char* argv[]) {
  /* command line arguments */
  int SEED;
  real_t T;
  unsigned P, K;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. trajectories")
    (",K", po::value(&K), "no. points to output")
    (",T", po::value(&T), "simulation time for each trajectory")
    ("seed", po::value(&SEED)->default_value(time(NULL)), "pseudorandom number seed")
    ("output", po::value(&OUTPUT)->default_value(false), "enable output")
    ("time", po::value(&TIME)->default_value(false), "enable timing output");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 0;
  }

  srand(SEED);
  int elapsed = simulate(P,K,T,OUTPUT);
  if (TIME) {
    std::cout << " " << elapsed;
  }

  return 0;
}
