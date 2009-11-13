/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Re$
 * $Date$
 */
#ifndef PRIOR_HPP
#define PRIOR_HPP

#include "model/NPZDModel.hpp"

#include "bi/pdf/MultiplicativeLogNormalPdf.hpp"
#include "bi/pdf/AdditiveGaussianPdf.hpp"
#include "bi/pdf/ConditionalFactoredPdf.hpp"

/*
 * Type list for proposal over parameters.
 */
BEGIN_TYPELIST(proposalP)
SINGLE_TYPE(2, bi::MultiplicativeLogNormalPdf<>)
SINGLE_TYPE(1, bi::AdditiveGaussianPdf<>)
SINGLE_TYPE(14, bi::MultiplicativeLogNormalPdf<>)
END_TYPELIST()

/**
 * Build proposal for p-nodes.
 */
bi::ConditionalFactoredPdf<GET_TYPELIST(proposalP)> buildPProposal(
    NPZDModel& m, const double scale);

#endif
