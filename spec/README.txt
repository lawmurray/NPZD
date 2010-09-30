This directory contains biogeochemical model specifications in the form of CSV
files, which can be readily edited by any spreadsheet program.

Model specifications are as follows:

* PZ.csv
  Lotka-Volterra PZ model.

* DeterministicNPZD.csv
  Deterministic NPZD model.

* StochasticNPZD.csv
  Stochastic NPZD model.

* GibbsNPZD.csv
  StochasticNPZD.csv with some parameters tweaked to make Gibbs updates of
  them easier, in particular moving mean parameters into exponents.

* UKFNPZD.csv
  StochasticNPZD.csv with parameters turned into discrete-time variables for
  online estimation with the unscented Kalman filter.

* UKFNPZD_Fixed*.csv
  Variants of UKFNPZD.csv with various parameters fixed.

To use, create a symlink NPZD.csv to any of these, run ./bootstrap.sh in the
main directory and compile following the instructions there.
