/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "prior.hpp"

#include "bi/pdf/ConditionalFactoredPdf.ipp"
#include "bi/pdf/FactoredPdf.ipp"

using namespace bi;

template class ConditionalFactoredPdf<GET_TYPELIST(proposalP)>;

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
