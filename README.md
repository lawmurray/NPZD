Synopsis
--------

    ./run.sh

This samples from the prior and posterior distributions and performs a
forecast. The `oct/` directory contains a few functions for plotting these
results (GNU Octave and OctBi required).


Description
-----------

This package is based on a single-box NPZD (nutrient, phytoplankton,
zooplankton and detritus) marine biogeochemical model. It consists of 15
parameters and 15 state variables. Four of the state variables (the $N$, $P$,
$Z$ and $D$) interact via a system of differential equations, with flux
between them determined by various nonlinear processes computed from the
remaining state variables, each of which is allowed to vary over time
following a first-order stochastic autoregressive process.

This version of the model was introduced in Parslow et al. (2013). Its
behaviour under sampling with the particle marginal Metropolis-Hastings (PMMH)
sampler was studied in Murray, Jones & Parslow (2013), then further studied
using a bridge particle filter in Del Moral & Murray (2014).

A real data set is provided from the site of Ocean Station Papa (OSP), taken
from Matear (1995).


References
----------

Parslow, J.; Cressie, N.; Campbell, E. P.; Jones, E. & Murray, L. M. Bayesian
Learning and Predictability in a Stochastic Nonlinear Dynamical
Model. *Ecological Applications*, 2013, 23, 679-698.

Murray, L. M.; Jones, E. M. & Parslow, J. On collapsed state-space models and
the particle marginal Metropolis-Hastings sampler, 2013.

Del Moral, P. & Murray, L. M. Sequential Monte Carlo with Highly Informative
Observations, 2014.

Matear, R. Parameter optimization and analysis of ecosystem models using
simulated annealing: A case study at Station P. *Journal of Marine Research*,
1995, 53, 571-607.
