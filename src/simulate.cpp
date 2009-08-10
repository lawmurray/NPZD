/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "bi/cuda/cuda.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <string>

extern void simulate(const unsigned P, const unsigned K, const real_t T,
    const int SEED, const std::string& INPUT_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME);

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;

  /* handle command line arguments */
  real_t T;
  unsigned P, K;
  int SEED;
  std::string INPUT_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. trajectories")
    (",K", po::value(&K), "no. points to output")
    (",T", po::value(&T), "simulation time for each trajectory")
    ("seed", po::value(&SEED)->default_value(time(NULL)), "pseudorandom number seed")
    ("input-file", po::value(&INPUT_FILE), "input file containing forcings")
    ("output-file", po::value(&OUTPUT_FILE), "output file to contain results")
    ("output", po::value(&OUTPUT)->default_value(false), "enable output")
    ("time", po::value(&TIME)->default_value(false), "enable timing output");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 0;
  }

  /* run simulation */
  simulate(P, K, T, SEED, INPUT_FILE, OUTPUT_FILE, OUTPUT, TIME);

  return 0;
}
