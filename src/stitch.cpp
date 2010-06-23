/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/buffer/ParticleMCMCNetCDFBuffer.hpp"

#include "boost/program_options.hpp"
#include "boost/typeof/typeof.hpp"
#include "boost/mpi.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <sys/time.h>

namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;
namespace mpi = boost::mpi;

using namespace bi;

int main(int argc, char* argv[]) {
  /* mpi */
  mpi::environment env(argc, argv);

  /* handle command line arguments */
  unsigned C, B, I;
  int SEED;
  std::vector<std::string> INPUT_FILES;
  std::string OUTPUT_FILE;

  po::options_description desc("Options");
  desc.add_options()
    (",C", po::value(&C)->default_value(1000), "no. samples to draw")
    (",B", po::value(&B)->default_value(0),
        "burn in to remove from each input sequence")
    (",I", po::value(&I)->default_value(100), "output interval")
    ("seed", po::value(&SEED)->default_value(0),
        "pseudorandom number seed")
    ("output-file", po::value(&OUTPUT_FILE),
        "output file to contain results")
    ("input-file", po::value(&INPUT_FILES),
        "input file containing proposals")
    ("help", "produce help message");
  po::positional_options_description pd;
  pd.add("input-file", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).
      run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  BI_ERROR(INPUT_FILES.size() > 0, "No input files specified");

  /* init stuff */
  bi_omp_init();
  NcError ncErr(NcError::silent_nonfatal);

  /* random number generator */
  Random rng(SEED);

  /* model */
  NPZDModel<> m;

  /* prior */
  NPZDPrior prior(m);

  /* ins */
  std::vector<ParticleMCMCNetCDFBuffer*> ins;
  unsigned i;
  for (i = 0; i < INPUT_FILES.size(); ++i) {
    ins.push_back(new ParticleMCMCNetCDFBuffer(m, INPUT_FILES[i]));
  }
  const unsigned T = ins[0]->size2();

  /* output */
  ParticleMCMCNetCDFBuffer out(m, C, T, OUTPUT_FILE, NetCDFBuffer::REPLACE);

  /* and go... */
  real t, l1, l2, p1, p2, a, lr, pr, i1, i2, j;
  host_matrix<> xd(m.getNetSize(D_NODE), T);
  host_matrix<> xc(m.getNetSize(C_NODE), T);
  host_vector<> theta(m.getNetSize(P_NODE));
  unsigned c, n, ch1, ch2, accepted = 0;
  bool accept;

  /* read log-likelihoods and priors into memory */
  const unsigned P = ins[0]->size1() - B;
  host_matrix<> ls(P, ins.size()), ps(P, ins.size());
  for (ch1 = 0; ch1 < ins.size(); ++ch1) {
    for (i1 = 0; i1 < P; ++i1) {
      ins[ch1]->readLogLikelihood(i1 + B, ls(i1,ch1));
      ins[ch1]->readPrior(i1 + B, ps(i1,ch1));
    }
  }

  /* write times */
  for (n = 0; n < T; ++n) {
    ins[0]->readTime(n, t);
    out.writeTime(n, t);
  }

  for (c = 0; c < C*I; ++c) {
    /* first select a chain, and then select a sample */
    ch2 = rng.uniformInt(0, ls.size2() - 1);
    i2 = rng.uniformInt(0, ls.size1() - 1);

    l2 = ls(i2,ch2);
    p2 = ps(i2,ch2);

    if (c == 0) {
      accept = true;
    } else {
      a = rng.uniform(0.0, 1.0);
      lr = exp(l2 - l1);
      pr = p2 / p1;
      accept = a < lr*pr;
    }

    if (accept) {
      ch1 = ch2;
      i1 = i2;
      l1 = l2;
      p1 = p2;
      ++accepted;
    }

    /* output */
    if (c % I == 0) {
      j = c / I;
      ins[ch1]->traceParticle(i1 + B, xd, xc);
      ins[ch1]->readSample(i1 + B, theta.buf());

      out.writeSample(j, theta.buf());
      out.writeLogLikelihood(j, l1);
      out.writePrior(j, p1);
      out.writeParticle(j, xd, xc);
      for (t = 0; t < T; ++t) {
        out.writeAncestor(t, j, j);
      }

      /* progress update */
      std::cerr << j << ' ';
    }

    /* progress update */
//    std::cerr << c << ":\t";
//    std::cerr.width(10);
//    std::cerr << l1;
//    std::cerr << '\t';
//    std::cerr.width(10);
//    std::cerr << p1;
//    std::cerr << "\tbeats\t";
//    std::cerr.width(10);
//    std::cerr << l2;
//    std::cerr << '\t';
//    std::cerr.width(10);
//    std::cerr << p2;
//    std::cerr << '\t';
//    if (accept) {
//      std::cerr << "accept";
//    }
//    std::cerr << std::endl;
  }
  std::cerr << std::endl;

  /* output diagnostics */
  std::cout << accepted << " of " << C*I << " proposals accepted" << std::endl;

  /* clean up */
  for (i = 0; i < ins.size(); ++i) {
    delete ins[i];
  }

  return 0;
}
