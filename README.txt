Requirements
------------------------------------------------------------------------------

The spec2x (code generation, graph generation) utility requires the
following packages:

* bi (the library code accompanying this)
* graphviz
* dot2tex
* sqlite3
* pdflatex
* DBI, DBD::SQLite, Text::CSV and List::MoreUtils Perl modules.

graphviz, sqlite3 and pdflatex are commonly available through Linux package
managers. dot2tex is available from <http://www.fauskes.net/code/dot2tex/>.
Perl modules may be installed using the cpan utility, also generally
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

perl Makefile.pl > Makefile

to generate a Makefile for the project. The following commands may then be
used:

make         : builds the simulate program in build/simulate,
make sql     : builds an SQLite database from the model spec, can be used to
               validate spec,
make dot     : builds the model graph in build/NPZD.pdf,
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
