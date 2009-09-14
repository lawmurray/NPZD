/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "bi/cuda/cuda.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <string>

extern void simulate(const unsigned P, const unsigned K, const real_t T,
    const unsigned NS, const int SEED, const std::string& INIT_FILE,
    const std::string& FORCE_FILE, const std::string& OBS_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME);

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;

  /* handle command line arguments */
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

  /* run simulation */
  simulate(P, K, T, NS, SEED, INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE,
      OUTPUT, TIME);

  return 0;
}
