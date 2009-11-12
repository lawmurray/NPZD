/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#ifndef MCMC_HPP
#define MCMC_HPP

#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"
#include "prior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/cuda/ode/IntegratorConstants.hpp"
#include "bi/state/State.hpp"
#include "bi/pdf/ConditionalFactoredPdf.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/MCMCNetCDFWriter.hpp"

/**
 * Initialise MCMC process.
 *
 * @param m Model.
 * @param x0 Prior.
 * @param q Proposal.
 * @param s State.
 * @param rng Random number generator.
 * @param fUpdater Updater for f-nodes.
 * @param oUpdater Updater for o-nodes.
 */
void init(NPZDModel& m, NPZDPrior& x0,
    bi::ConditionalFactoredPdf<GET_TYPELIST(proposalP)>& q, bi::State& s,
    bi::Random& rng, bi::FUpdater* fUpdater, bi::OUpdater* oUpdater);

/**
 * Take one step of chain.
 *
 * @param T Time to filter.
 * @param minEss ESS threshold.
 * @param lambda Temperature.
 * @param[out] theta State.
 * @param[out] l Log-likelihood of state.
 *
 * @return True if proposal accepted, false otherwise.
 */
bool step(const real_t T, const real_t minEss, const double lambda,
    bi::state_vector& theta, double& l);

/**
 * Destroy MCMC process.
 */
void destroy();

#endif
