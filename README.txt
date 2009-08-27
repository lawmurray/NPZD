NPZD model implementation and spec2x scripts
------------------------------------------------------------------------------

Implementation of the NPZD model, and spec2x scripts for graph and code generation.


Requirements
------------------------------------------------------------------------------

The code requires the following:

  * The bi library, and its requirements, see its README.txt file.
  * The Boost.ProgramOptions library in addition to these requirements.

The spec2x utility requires the following:

  * sqlite3,

and the following Perl modules:

  * DBI,
  * DBD::SQLite,
  * Text::CSV,
  * List::MoreUtils.

The following is optional for building source code documentation:

  * Doxygen <www.doxygen.org>,
  
and the following for the model visualisation:

  * pdflatex,
  * graphviz,
  * dot2tex <www.fauskes.net/code/dot2tex/>.

graphviz, sqlite3 and pdflatex are commonly available through Linux package
managers. Perl modules may be installed using the cpan utility, also generally
available in Linux package managers. Run cpan from the command line, then:

install DBI
install DBD::SQLite
...etc


Specification
------------------------------------------------------------------------------

The model specification is given in spec/NPZD.csv. C++ code for the model is
automatically generated from this specification using the spec2x Perl scripts.


Compiling
------------------------------------------------------------------------------

Run:

./bootstrap.sh

to generate a Makefile for the project. The following commands may then be
used:

make         : builds the simulate program in build/simulate,
make sql     : builds an SQLite database from the model spec, can be used to
               validate spec,
make dot     : builds the model graph in build/NPZD.pdf,
make cpp     : generates code for the model,
make clean   : cleans up all files.

The Perl script checks for all dependencies and incorporates them into the
Makefile. Note that it does not compile the bi library, however. The bi
library should be recompiled separately when changes are made to it. This is
automatic in Eclipse, as it is registered as a project dependency.

The following macros may be defined during compilation to adjust behaviour,
set these with, e.g. 'make USE_DOUBLE=1'

* NDEBUG will disable assertion checking.
* USE_DOUBLE will use double precision arithmetic on the GPU, otherwise single
  precision is used.
* USE_FAST_MATH will use intrinsic CUDA functions throughout, as long as
  USE_DOUBLE is not defined (intrinsics are available only for single
  precision). No intrinsics are used by default.
* USE_DOPRI5 will use the DOPRI5 integrator for ordinary differential
  equations, otherwise RK4(3)5[2R+]C is used.
