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

#include "bi/pdf/LogNormalPdf.hpp"
#include "bi/pdf/GaussianPdf.hpp"
#include "bi/pdf/FactoredPdf.hpp"
#include "bi/pdf/MultiplicativeLogNormalPdf.hpp"
#include "bi/pdf/AdditiveGaussianPdf.hpp"
#include "bi/pdf/ConditionalFactoredPdf.hpp"
#include "bi/math/vector.hpp"
#include "bi/math/matrix.hpp"

/*
 * Type list for prior over parameters.
 */
BEGIN_TYPELIST(priorP)
SINGLE_TYPE(2, bi::LogNormalPdf<>)
SINGLE_TYPE(1, bi::GaussianPdf<>)
SINGLE_TYPE(14, bi::LogNormalPdf<>)
END_TYPELIST()

/*
 * Type list for proposal over parameters.
 */
BEGIN_TYPELIST(proposalP)
SINGLE_TYPE(2, bi::MultiplicativeLogNormalPdf<>)
SINGLE_TYPE(1, bi::AdditiveGaussianPdf<>)
SINGLE_TYPE(14, bi::MultiplicativeLogNormalPdf<>)
END_TYPELIST()

/**
 * Build prior for p-nodes.
 */
bi::FactoredPdf<GET_TYPELIST(priorP)> buildPPrior(NPZDModel& m);

/**
 * Build prior for p-nodes.
 */
bi::ConditionalFactoredPdf<GET_TYPELIST(proposalP)> buildPProposal(
    NPZDModel& m, const double scale);

/**
 * Build prior for d-nodes.
 */
bi::LogNormalPdf<bi::vector,bi::diagonal_matrix> buildDPrior(NPZDModel& m);

/**
 * Build prior for c-nodes.
 */
bi::LogNormalPdf<bi::vector,bi::diagonal_matrix> buildCPrior(NPZDModel& m);

#endif
