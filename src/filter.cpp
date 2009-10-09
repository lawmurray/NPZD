/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#include "filter.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"

#include <iostream>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;

using namespace bi;

int main(int argc, char* argv[]) {
  /* handle command line arguments */
  real_t T, h;
  unsigned P, K, INIT_NS, FORCE_NS, OBS_NS;
  int SEED;
  std::string INIT_FILE, FORCE_FILE, OBS_FILE, OUTPUT_FILE;
  bool OUTPUT, TIME;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    (",P", po::value(&P), "no. particles")
    (",K", po::value(&K), "size of intermediate result buffer")
    (",T", po::value(&T), "simulation time for each trajectory")
    (",h", po::value(&h)->default_value(0.2),
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

  /* random number generator */
  Random rng(SEED);

  /* report missing variables in NetCDF, but don't die */
  #ifndef NDEBUG
  NcError ncErr(NcError::verbose_nonfatal);
  #else
  NcError ncErr(NcError::silent_nonfatal);
  #endif

  /* model */
  NPZDModel m;

  /* state and intermediate results */
  State s(m, P);
  //Result<real_t> r(m, P, K);

  /* initialise from file... */
  NetCDFReader<real_t,true,true,true,false,true,true,true> in(m, INIT_FILE, INIT_NS);
  in.read(s);

  /* ...and/or initialise from prior */
  //BOOST_AUTO(p0, buildPPrior(m));
  BOOST_AUTO(d0, buildDPrior(m));
  BOOST_AUTO(c0, buildCPrior(m));

  //p0.sample(s.pState, rng);
  d0.sample(s.dState, rng);
  c0.sample(s.cState, rng);

  /* forcings, observations */
  FUpdater<> fUpdater(m, FORCE_FILE, s, FORCE_NS);
  OUpdater<> oUpdater(m, OBS_FILE, s, OBS_NS);

  /* output */
  NetCDFWriter<>* out;
  if (OUTPUT) {
    out = new NetCDFWriter<>(m, OUTPUT_FILE, P, oUpdater.numUniqueTimes() + 1);
  } else {
    out = NULL;
  }

  /* filter */
  timeval start, end;
  gettimeofday(&start, NULL);

  filter(T, h, m, s, rng, NULL, &fUpdater, &oUpdater, out);

  gettimeofday(&end, NULL);
  if (TIME) {
    int elapsed = (end.tv_sec-start.tv_sec)*1e6 + (end.tv_usec-start.tv_usec);
    std::cout << elapsed << std::endl;
  }

  return 0;
}

LogNormalPdf<vector,diagonal_matrix> buildPPrior(NPZDModel& m) {
  const unsigned N = m.getPSize();

  vector mu(N);
  diagonal_matrix sigma(N,N);
  BOOST_AUTO(sigmad, diag(sigma));

  mu[m.getPNode("KW")->getId()] = log(0.03);
  mu[m.getPNode("KC")->getId()] = log(0.04);
  mu[m.getPNode("deltaS")->getId()] = log(5.0); // should be Gaussian!
  mu[m.getPNode("deltaI")->getId()] = log(0.5);
  mu[m.getPNode("P_DF")->getId()] = log(0.4);
  mu[m.getPNode("Z_DF")->getId()] = log(0.4);
  mu[m.getPNode("alphaC")->getId()] = log(1.2);
  mu[m.getPNode("alphaCN")->getId()] = log(0.4);
  mu[m.getPNode("alphaCh")->getId()] = log(0.03);
  mu[m.getPNode("alphaA")->getId()] = log(0.3);
  mu[m.getPNode("alphaNC")->getId()] = log(0.25);
  mu[m.getPNode("alphaI")->getId()] = log(4.7);
  mu[m.getPNode("alphaCl")->getId()] = log(0.2);
  mu[m.getPNode("alphaE")->getId()] = log(0.32);
  mu[m.getPNode("alphaR")->getId()] = log(0.1);
  mu[m.getPNode("alphaQ")->getId()] = log(0.01);
  mu[m.getPNode("alphaL")->getId()] = log(0.0);

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

  return LogNormalPdf<vector,diagonal_matrix>(mu, sigma);
}

LogNormalPdf<vector,diagonal_matrix> buildDPrior(NPZDModel& m) {
  const unsigned N = m.getDSize();

  vector mu(N);
  diagonal_matrix sigma(N,N);
  BOOST_AUTO(sigmad, diag(sigma));

  mu[m.getDNode("muC")->getId()] = log(1.2);
  mu[m.getDNode("muCN")->getId()] = log(0.4);
  mu[m.getDNode("muCh")->getId()] = log(0.033);
  mu[m.getDNode("nuA")->getId()] = log(0.3);
  mu[m.getDNode("piNC")->getId()] = log(0.25);
  mu[m.getDNode("zetaI")->getId()] = log(4.7);
  mu[m.getDNode("zetaCl")->getId()] = log(0.2);
  mu[m.getDNode("zetaE")->getId()] = log(0.32);
  mu[m.getDNode("nuR")->getId()] = log(0.1);
  mu[m.getDNode("zetaQ")->getId()] = log(0.01);
  mu[m.getDNode("zetaL")->getId()] = log(0.0);
  mu[m.getDNode("Chla")->getId()] = log(0.28);

  sigmad[m.getDNode("muC")->getId()] = 0.1;
  sigmad[m.getDNode("muCN")->getId()] = 0.1;
  sigmad[m.getDNode("muCh")->getId()] = 0.1;
  sigmad[m.getDNode("nuA")->getId()] = 0.1;
  sigmad[m.getDNode("piNC")->getId()] = 0.1;
  sigmad[m.getDNode("zetaI")->getId()] = 0.1;
  sigmad[m.getDNode("zetaCl")->getId()] = 0.1;
  sigmad[m.getDNode("zetaE")->getId()] = 0.1;
  sigmad[m.getDNode("nuR")->getId()] = 0.1;
  sigmad[m.getDNode("zetaQ")->getId()] = 0.1;
  sigmad[m.getDNode("zetaL")->getId()] = 0.1;
  sigmad[m.getDNode("Chla")->getId()] = 0.1;

  return LogNormalPdf<vector,diagonal_matrix>(mu, sigma);
}

LogNormalPdf<vector,diagonal_matrix> buildCPrior(NPZDModel& m) {
  const unsigned N = m.getCSize();

  vector mu(N);
  diagonal_matrix sigma(N,N);
  BOOST_AUTO(sigmad, diag(sigma));

  mu[m.getCNode("P")->getId()] = log(1.64);
  mu[m.getCNode("Z")->getId()] = log(1.91);
  mu[m.getCNode("D")->getId()] = log(1.3);
  mu[m.getCNode("N")->getId()] = log(9.3);

  sigmad[m.getCNode("P")->getId()] = 0.1;
  sigmad[m.getCNode("Z")->getId()] = 0.1;
  sigmad[m.getCNode("D")->getId()] = 0.1;
  sigmad[m.getCNode("N")->getId()] = 0.1;

  return LogNormalPdf<vector,diagonal_matrix>(mu, sigma);
}
