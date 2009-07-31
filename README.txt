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
