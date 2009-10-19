/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "prior.hpp"

using namespace bi;

FactoredPdf<GET_TYPELIST(priorP)> buildPPrior(NPZDModel& m) {
  FactoredPdf<GET_TYPELIST(priorP)> p;

  p.set(m.KW.id, LogNormalPdf<>(log(0.03), pow(0.2,2.0)));
  p.set(m.KC.id, LogNormalPdf<>(log(0.04), pow(0.3,2.0)));
  p.set(m.deltaS.id, GaussianPdf<>(5.0, pow(1.0,2.0)));
  p.set(m.deltaI.id, LogNormalPdf<>(log(0.5), pow(0.1,2.0)));
  p.set(m.P_DF.id, LogNormalPdf<>(log(0.4), pow(0.2,2.0)));
  p.set(m.Z_DF.id, LogNormalPdf<>(log(0.4), pow(0.1,2.0)));
  p.set(m.alphaC.id, LogNormalPdf<>(log(1.2), pow(0.63,2.0)));
  p.set(m.alphaCN.id, LogNormalPdf<>(log(0.4), pow(0.2,2.0)));
  p.set(m.alphaCh.id, LogNormalPdf<>(log(0.03), pow(0.37,2.0)));
  p.set(m.alphaA.id, LogNormalPdf<>(log(0.3), pow(1.0,2.0)));
  p.set(m.alphaNC.id, LogNormalPdf<>(log(0.25), pow(0.3,2.0)));
  p.set(m.alphaI.id, LogNormalPdf<>(log(4.7), pow(0.7,2.0)));
  p.set(m.alphaCl.id, LogNormalPdf<>(log(0.2), pow(1.3,2.0)));
  p.set(m.alphaE.id, LogNormalPdf<>(log(0.32), pow(0.25,2.0)));
  p.set(m.alphaR.id, LogNormalPdf<>(log(0.1), pow(0.5,2.0)));
  p.set(m.alphaQ.id, LogNormalPdf<>(log(0.01), pow(1.0,2.0)));
  p.set(m.alphaL.id, LogNormalPdf<>(log(0.01), pow(0.1,2.0)));

  return p;
}

ConditionalFactoredPdf<GET_TYPELIST(proposalP)> buildPProposal(NPZDModel& m,
    const double scale) {
  ConditionalFactoredPdf<GET_TYPELIST(proposalP)> p;

  p.set(m.KW.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.KC.id, MultiplicativeLogNormalPdf<>(pow(scale*0.3,2.0)));
  p.set(m.deltaS.id, AdditiveGaussianPdf<>(pow(scale*1.0,2.0)));
  p.set(m.deltaI.id, MultiplicativeLogNormalPdf<>(pow(scale*0.1,2.0)));
  p.set(m.P_DF.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.Z_DF.id, MultiplicativeLogNormalPdf<>(pow(scale*0.1,2.0)));
  p.set(m.alphaC.id, MultiplicativeLogNormalPdf<>(pow(scale*0.63,2.0)));
  p.set(m.alphaCN.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.alphaCh.id, MultiplicativeLogNormalPdf<>(pow(scale*0.37,2.0)));
  p.set(m.alphaA.id, MultiplicativeLogNormalPdf<>(pow(scale*1.0,2.0)));
  p.set(m.alphaNC.id, MultiplicativeLogNormalPdf<>(pow(scale*0.3,2.0)));
  p.set(m.alphaI.id, MultiplicativeLogNormalPdf<>(pow(scale*0.7,2.0)));
  p.set(m.alphaCl.id, MultiplicativeLogNormalPdf<>(pow(scale*1.3,2.0)));
  p.set(m.alphaE.id, MultiplicativeLogNormalPdf<>(pow(scale*0.25,2.0)));
  p.set(m.alphaR.id, MultiplicativeLogNormalPdf<>(pow(scale*0.5,2.0)));
  p.set(m.alphaQ.id, MultiplicativeLogNormalPdf<>(pow(scale*1.0,2.0)));
  p.set(m.alphaL.id, MultiplicativeLogNormalPdf<>(pow(scale*0.1,2.0)));

  return p;
}

GaussianPdf<zero_vector,diagonal_matrix> buildSPrior(NPZDModel& m) {
  const unsigned N = m.getSSize();

  zero_vector mu(N);
  diagonal_matrix sigma(N,N);
  sigma.clear();

  return GaussianPdf<zero_vector,diagonal_matrix>(mu, sigma);
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
