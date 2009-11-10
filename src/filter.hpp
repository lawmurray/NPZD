/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev:234 $
 * $Date:2009-08-17 11:10:31 +0800 (Mon, 17 Aug 2009) $
 */
#ifndef FILTER_HPP
#define FILTER_HPP

#include "model/NPZDModel.hpp"
#include "model/NPZDPrior.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/random/Random.hpp"
#include "bi/method/RUpdater.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFWriter.hpp"

/**
 * Run particle filter.
 *
 * @param T Time to filter.
 * @param h Initial step size.
 * @param m Model.
 * @param s State.
 * @param rng Random number generator.
 * @param r Intermediate results.
 * @param fUpdater Updater for f-nodes.
 * @param oUpdater Updater for o-nodes.
 * @param out Output, NULL for no output.
 */
void filter(const real_t T, const real_t h, NPZDModel& m, bi::State& s,
    bi::Random& rng, bi::FUpdater<>* fUpdater, bi::OUpdater<>* oUpdater,
    bi::NetCDFWriter<>* out = NULL);

#endif
