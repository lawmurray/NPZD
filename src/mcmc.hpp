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

#include "bi/cuda/cuda.hpp"
#include "bi/state/State.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFWriter.hpp"

/**
 * Initialise GPU particle filter.
 *
 * @param h Initial step size.
 * @param m Model.
 * @param s State.
 * @param rng Random number generator.
 * @param fUpdater Updater for f-nodes.
 * @param oUpdater Updater for o-nodes.
 */
void init(const real_t h, NPZDModel& m, bi::State& s, bi::Random& rng,
    bi::FUpdater<>* fUpdater, bi::OUpdater<>* oUpdater);

/**
 * Run particle filter to calculate log-likelihood.
 *
 * @param T Time to filter.
 * @param minEss ESS threshold.
 *
 * @return Log-likelihood.
 */
real_t filter(const real_t T, const real_t minEss);

/**
 * Destroy particle filter.
 */
void destroy();

#endif
