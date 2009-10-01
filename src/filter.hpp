#ifndef FILTER_HPP
#define FILTER_HPP

#include "model/NPZDModel.hpp"

#include "bi/cuda/cuda.hpp"
#include "bi/random/Random.hpp"
#include "bi/pdf/LogNormalPdf.hpp"
#include "bi/method/FUpdater.hpp"
#include "bi/method/OUpdater.hpp"
#include "bi/io/NetCDFWriter.hpp"

/**
 * Build prior for p-nodes.
 */
bi::LogNormalPdf<bi::vector,bi::diagonal_matrix> buildPPrior(NPZDModel& m);

/**
 * Build prior for d-nodes.
 */
bi::LogNormalPdf<bi::vector,bi::diagonal_matrix> buildDPrior(NPZDModel& m);

/**
 * Build prior for c-nodes.
 */
bi::LogNormalPdf<bi::vector,bi::diagonal_matrix> buildCPrior(NPZDModel& m);

/**
 * Run particle filter.
 *
 * @param T Time to filter.
 * @param m Model.
 * @param s State.
 * @param rng Random number generator.
 * @param r Intermediate results.
 * @param fUpdater Updater for f-nodes.
 * @param oUpdater Updater for o-nodes.
 * @param out Output, NULL for no output.
 */
void filter(const real_t T, NPZDModel& m, bi::State& s, bi::Random& rng,
    bi::Result<>* r, bi::FUpdater<>* fUpdater, bi::OUpdater<>* oUpdater,
    bi::NetCDFWriter<>* out = NULL);

#endif
