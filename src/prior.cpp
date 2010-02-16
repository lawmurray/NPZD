/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "prior.hpp"

using namespace bi;

ConditionalFactoredPdf<GET_TYPELIST(proposalP)> buildPProposal(NPZDModel& m,
    const double scale) {
  ConditionalFactoredPdf<GET_TYPELIST(proposalP)> p;

  p.set(m.Kw.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.KCh.id, MultiplicativeLogNormalPdf<>(pow(scale*0.3,2.0)));
  p.set(m.Dsi.id, AdditiveGaussianPdf<>(pow(scale*1.0,2.0)));
  p.set(m.ZgD.id, MultiplicativeLogNormalPdf<>(pow(scale*0.1,2.0)));
  p.set(m.PDF.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.ZDF.id, MultiplicativeLogNormalPdf<>(pow(scale*0.1,2.0)));
  p.set(m.muPgC.id, MultiplicativeLogNormalPdf<>(pow(scale*0.63,2.0)));
  p.set(m.muPgR.id, MultiplicativeLogNormalPdf<>(pow(scale*0.2,2.0)));
  p.set(m.muPCh.id, MultiplicativeLogNormalPdf<>(pow(scale*0.37,2.0)));
  p.set(m.muPaN.id, MultiplicativeLogNormalPdf<>(pow(scale*1.0,2.0)));
  p.set(m.muPgR.id, MultiplicativeLogNormalPdf<>(pow(scale*0.3,2.0)));
  p.set(m.muZin.id, MultiplicativeLogNormalPdf<>(pow(scale*0.7,2.0)));
  p.set(m.muZCl.id, MultiplicativeLogNormalPdf<>(pow(scale*1.3,2.0)));
  p.set(m.muZgE.id, MultiplicativeLogNormalPdf<>(pow(scale*0.25,2.0)));
  p.set(m.muDre.id, MultiplicativeLogNormalPdf<>(pow(scale*0.5,2.0)));
  p.set(m.muZmQ.id, MultiplicativeLogNormalPdf<>(pow(scale*1.0,2.0)));

  return p;
}
