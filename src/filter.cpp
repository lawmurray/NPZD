/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/pdf/LogNormalPdf.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <string>

namespace po = boost::program_options;

using namespace bi;

LogNormalPdf<vector,banded_matrix> buildPPrior(NPZDModel& m);

extern void filter(const unsigned P, const unsigned K, const real_t T,
    const unsigned NS, const int SEED, const std::string& INIT_FILE,
    const std::string& FORCE_FILE, const std::string& OBS_FILE,
    const std::string& OUTPUT_FILE, const bool OUTPUT, const bool TIME);

int main(int argc, char* argv[]) {
  /* handle command line arguments */
  real_t T;
  unsigned P, K, NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. particles")
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

  /* priors */
  //BOOST_AUTO(p0, buildPPrior(m));
  //BOOST_AUTO(buildDPrior(m), d0);
  //BOOST_AUTO(buildCPrior(m), c0);

  //p0.sample(s.pState, rng);
  //d0.sample(s.dState, rng);
  //c0.sample(s.cState, rng);

  /* run filter */
  filter(P, K, T, NS, SEED, INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE, OUTPUT, TIME);

  return 0;
}

LogNormalPdf<vector,banded_matrix> buildPPrior(NPZDModel& m) {
  const unsigned N = m.getPSize();

  vector mu(N);
  banded_matrix sigma(N,N);
  BOOST_AUTO(sigmad, diag(sigma));

  mu[m.getPNode("KW")->getId()] = 0.03;
  mu[m.getPNode("KC")->getId()] = 0.04;
  mu[m.getPNode("deltaS")->getId()] = 5.0; // should be Gaussian!
  mu[m.getPNode("deltaI")->getId()] = 0.5;
  mu[m.getPNode("P_DF")->getId()] = 0.4;
  mu[m.getPNode("Z_DF")->getId()] = 0.4;
  mu[m.getPNode("alphaC")->getId()] = 1.2;
  mu[m.getPNode("alphaCN")->getId()] = 0.4;
  mu[m.getPNode("alphaCh")->getId()] = 0.03;
  mu[m.getPNode("alphaA")->getId()] = 0.3;
  mu[m.getPNode("alphaNC")->getId()] = 0.25;
  mu[m.getPNode("alphaI")->getId()] = 4.7;
  mu[m.getPNode("alphaCl")->getId()] = 0.2;
  mu[m.getPNode("alphaE")->getId()] = 0.32;
  mu[m.getPNode("alphaR")->getId()] = 0.1;
  mu[m.getPNode("alphaQ")->getId()] = 0.01;
  mu[m.getPNode("alphaL")->getId()] = 0.0;

  sigmad[m.getPNode("KW")->getId()] = 0.2;
  sigmad[m.getPNode("KC")->getId()] = 0.3;
  sigmad[m.getPNode("deltaS")->getId()] = 1.0;
  sigmad[m.getPNode("deltaI")->getId()] = 0.1;
  sigmad[m.getPNode("P_DF")->getId()] = 0.2;
  sigmad[m.getPNode("Z_DF")->getId()] = 0.2;
  sigmad[m.getPNode("alphaC")->getId()] = 0.63;
  sigmad[m.getPNode("alphaCN")->getId()] = 0.2;
  sigmad[m.getPNode("alphaCh")->getId()] = 0.37;
  sigmad[m.getPNode("alphaA")->getId()] = 1.0;
  sigmad[m.getPNode("alphaNC")->getId()] = 0.3;
  sigmad[m.getPNode("alphaI")->getId()] = 0.7;
  sigmad[m.getPNode("alphaCl")->getId()] = 1.3;
  sigmad[m.getPNode("alphaE")->getId()] = 0.25;
  sigmad[m.getPNode("alphaR")->getId()] = 0.5;
  sigmad[m.getPNode("alphaQ")->getId()] = 1.0;
  sigmad[m.getPNode("alphaL")->getId()] = 0.0;

  return LogNormalPdf<vector,banded_matrix>(mu, sigma);
}
