##
## Generate a Makefile for compilation.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

# Settings
$NAME = 'NPZD';
$SPEC = 'NPZD';
$SRCDIR = 'src';
$BUILDDIR = 'build';
$SPECDIR = 'spec';
$SPEC2XDIR = 'spec2x';
$SPEC2XSRCDIR = "$SPEC2XDIR/src";
$SPEC2XCPPTEMPLATEDIR = "$SPEC2XDIR/templates/cpp";
$SPEC2XTEXTEMPLATEDIR = "$SPEC2XDIR/templates/tex";
$CPPDIR = "$SRCDIR/model";

# Compile flags
$CXX = 'g++';
$CUDACC = 'nvcc';
$LINKER = 'nvcc';
$CXXFLAGS = '-Wall -I"../bi/src" `nc-config --cflags`';
$CUDACCFLAGS = '-arch=sm_13 -Xptxas="-v" -I"../bi/src" -I"$GSL_ROOT/include" `nc-config --cflags` -DBOOST_NO_INCLASS_MEMBER_INITIALIZATION -DBOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS -I/tools/thrust/1.1.1';
$LINKFLAGS = '-L"../bi/build" -lbi -latlas -lf77blas -llapack -lgfortran -lboost_program_options-gcc43-mt -lgslcblas -lgsl `nc-config --libs` -lnetcdf_c++';
# ^ may need f2c, g2c or nothing in place of gfortran
$DEPFLAGS = '-I"../bi/src"'; # flags for dependencies check

# Release flags
#$CXXFLAGS .= ' -O3 -funroll-loops -fomit-frame-pointer';
#$CUDACCFLAGS .= ' -O3 --compiler-options="-O3 -funroll-loops -fomit-frame-pointer"';

# Debugging flags
$CXXFLAGS .= ' -g';
$CUDACCFLAGS .= ' -g';

# Profiling flags
#$CXXFLAGS .= ' -pg';
#$CUDACCFLAGS .= ' --compiler-options="-pg"';

# Disassembly flags
#$CUDACCFLAGS .= ' -keep';

# Bootstrap
`mkdir -p $BUILDDIR $CPPDIR`;
`perl $SPEC2XSRCDIR/csv2sql.pl --model $NAME --outdir $BUILDDIR --srcdir $SPEC2XSRCDIR < $SPECDIR/$SPEC.csv`;
`perl $SPEC2XSRCDIR/sql2cpp.pl --outdir $CPPDIR --templatedir $SPEC2XCPPTEMPLATEDIR --model $NAME --dbfile $BUILDDIR/$NAME.db`;

# Walk through source
@files = ($SRCDIR);
while (@files) {
  $file = shift @files;
  if (-d $file) {
    # recurse into directory
    opendir(DIR, $file);
    push(@files, map { "$file/$_" } grep { !/^\./ } readdir(DIR));
    closedir(DIR);
  } elsif (-f $file && $file =~ /\.(?:cu|c|cpp)$/) {
    # determine compiler and appropriate flags
    if ($file =~ /\.cu$/) {
      $cc = $CUDACC;
      $ccstr = "\$(CUDACC)";
      $flags = $CUDACCFLAGS;
      $flagstr = "\$(CUDACCFLAGS)";
    } else {
      $cc = $CXX;
      $ccstr = "\$(CXX)";
      $flags = $CXXFLAGS;
      $flagstr = "\$(CXXFLAGS)";
    }
    
    # determine dependencies of this source and construct Makefile target
    $target = $file;
    $target =~ s/^$SRCDIR/$BUILDDIR/;
    $target =~ s/\.\w+$/.o/;

    $target =~ /(.*)\//;
    $dir = $1;
    $dirs{$dir} = 1;

    $command = "$dir/" . `$cc $flags -M $file`;
    chomp $command;
    $command .= " \\\n    $dir\n";
    $command .= "\t$ccstr -o $target $flagstr -c $file\n";
    $command .= "\trm -f *.linkinfo\n";
    push(@targets, $target);
    push(@commands, $command);
  }
}

# Write Makefile
print <<End;
NAME=$NAME
SPEC=$SPEC

BUILDDIR=$BUILDDIR
SRCDIR=$SRCDIR
SPECDIR=$SPECDIR
SPEC2XDIR=$SPEC2XDIR
SPEC2XSRCDIR=$SPEC2XSRCDIR
SPEC2XCPPTEMPLATEDIR=$SPEC2XCPPTEMPLATEDIR
SPEC2XTEXTEMPLATEDIR=$SPEC2XTEXTEMPLATEDIR
CPPDIR=$CPPDIR

CXX=$CXX
CXXFLAGS=$CXXFLAGS
LINKER=$LINKER
CUDACC=$CUDACC
CUDACCFLAGS=$CUDACCFLAGS
LINKFLAGS=$LINKFLAGS

ifdef USE_DOUBLE
CUDACCFLAGS += -DUSE_DOUBLE
CXXFLAGS += -DUSE_DOUBLE
else
ifdef USE_FAST_MATH
CUDACCFLAGS += -use_fast_math
endif
endif

ifdef USE_DOPRI5
CUDACCFLAGS += -DUSE_DOPRI5
CXXFLAGS += -DUSE_DOPRI5
endif

ifdef NDEBUG
CUDACCFLAGS += -DNDEBUG
CXXFLAGS += -DNDEBUG
endif

ifdef SIZE
CUDACCFLAGS += -DSIZE=\$(SIZE)
CXXFLAGS += -DSIZE=\$(SIZE)
endif

End

# Default targets
print <<End;
default: \$(BUILDDIR)/simulate \$(BUILDDIR)/filter

End

# spec2x targets
print <<End;

\$(BUILDDIR)/\$(NAME).db: \$(SPECDIR)/\$(SPEC).csv \$(SPEC2XSRCDIR)/csv2sql.pl \$(SPEC2XSRCDIR)/sqlite.sql
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XSRCDIR)/csv2sql.pl --model \$(NAME) --outdir \$(BUILDDIR) --srcdir \$(SPEC2XSRCDIR) < \$(SPECDIR)/\$(SPEC).csv 

\$(BUILDDIR)/\$(NAME).dot: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XSRCDIR)/sql2dot.pl
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XSRCDIR)/sql2dot.pl --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db > \$(BUILDDIR)/\$(NAME).dot

\$(BUILDDIR)/\$(NAME)_graph.tex: \$(BUILDDIR)/\$(NAME).dot
\tmkdir -p \$(BUILDDIR)
\tdot2tex --autosize --usepdflatex --figpreamble="\huge" --prog=neato -traw --crop < \$(BUILDDIR)/\$(NAME).dot > \$(BUILDDIR)/\$(NAME)_graph.tex

\$(BUILDDIR)/\$(NAME)_graph.pdf: \$(BUILDDIR)/\$(NAME)_graph.tex
\tmkdir -p \$(BUILDDIR)
\tcd \$(BUILDDIR); \\
\tpdflatex \$(NAME)_graph.tex; \\
\tpdflatex \$(NAME)_graph.tex; \\
\tcd ..

\$(BUILDDIR)/\$(NAME).tex: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XSRCDIR)/sql2tex.pl \$(SPEC2XTEXTEMPLATEDIR)/*.template
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XSRCDIR)/sql2tex.pl --templatedir \$(SPEC2XTEXTEMPLATEDIR) --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db > \$(BUILDDIR)/\$(NAME).tex

\$(BUILDDIR)/\$(NAME).pdf: \$(BUILDDIR)/\$(NAME).tex
\tmkdir -p \$(BUILDDIR)
\tcd \$(BUILDDIR); \\
\tpdflatex \$(NAME).tex; \\
\tpdflatex \$(NAME).tex; \\
\tcd ..

sql: \$(BUILDDIR)/\$(NAME).db

dot: \$(BUILDDIR)/\$(NAME)_graph.pdf

cpp: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XSRCDIR)/sql2cpp.pl \$(SPEC2XCPPTEMPLATEDIR)/*.template
\tmkdir -p \$(CPPDIR)
\tperl \$(SPEC2XSRCDIR)/sql2cpp.pl --outdir \$(CPPDIR) --templatedir \$(SPEC2XCPPTEMPLATEDIR) --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db

tex: \$(BUILDDIR)/\$(NAME).tex

pdf: \$(BUILDDIR)/\$(NAME).pdf

End

# Artefacts
print "\$(BUILDDIR)/simulate: build/simulate.o build/gpusimulate.o build/model/NPZDModel.o\n";
print "\t\$(LINKER) -o $BUILDDIR/simulate \$(LINKFLAGS) build/simulate.o build/gpusimulate.o build/model/NPZDModel.o\n\n";

print "\$(BUILDDIR)/filter: build/filter.o build/gpufilter.o build/model/NPZDModel.o\n";
print "\t\$(LINKER) -o $BUILDDIR/filter \$(LINKFLAGS) build/filter.o build/gpufilter.o build/model/NPZDModel.o\n\n";

# Targets
print join("\n", @commands);
print "\n";

# Build directory tree targets
foreach $dir (sort keys %dirs) {
  print "$dir:\n\tmkdir -p $dir\n\n";
}

# Clean target
print <<End;
clean:
\trm -rf \$(BUILDDIR)

End
