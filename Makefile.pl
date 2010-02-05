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
$SPEC2XDIR = '../spec2x';
$SPEC2XSRCDIR = "$SPEC2XDIR/src";
$SPEC2XCPPTTDIR = "$SPEC2XDIR/tt/cpp";
$SPEC2XTEXTEMPLATEDIR = "$SPEC2XDIR/templates/tex";
$CPPDIR = "$SRCDIR/model";

# Compile flags
$CXX = 'g++';
$CUDACC = 'nvcc';
$LINKER = 'g++';
$CPPINCLUDES = '-I../bi/src -I/usr/local/cuda/include -I/tools/cuda/2.3/cuda/include/ -I/tools/thrust/1.1.1 -I/usr/local/include/thrust -I/tools/magma/0.2/include';
$CXXFLAGS = "-Wall -fopenmp `nc-config --cflags` `mpic++ -showme:compile` $CPPINCLUDES";
$CUDACCFLAGS = "-arch=sm_13 --maxrregcount 64 -Xptxas=\"-v\" -Xcompiler=\"-Wall -fopenmp `mpic++ -showme:compile`\" `nc-config --cflags` -DBOOST_NO_INCLASS_MEMBER_INITIALIZATION -DBOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS $CPPINCLUDES";
$LINKFLAGS = '-L"../bi/build" -L"/usr/local/atlas/lib" -L"/tools/magma/0.2/lib" -lbi -latlas -lf77blas -lcblas -llapack -lmagma -lmagmablas -lgfortran -lgsl -lnetcdf_c++ `nc-config --libs` -lcuda -lgomp -lpthread -lboost_program_options-gcc43-mt -lboost_mpi-gcc43-mt `mpic++ -showme:link`';
# ^ may need f2c, g2c or nothing in place of gfortran
# ^ may need to add -lcuda as well as -lcudart
$DEPFLAGS = '-I"../bi/src"'; # flags for dependencies check

# Release flags
$RELEASE_CXXFLAGS = ' -O3 -funroll-loops -fomit-frame-pointer';
$RELEASE_CUDACCFLAGS = ' -O3 -Xcompiler="-O3 -funroll-loops -fomit-frame-pointer"';
$RELEASE_LINKFLAGS = ' -lcublas';
# ^ have observed errors in unscented Kalman filter with -O3 to nvcc, so using -O2

# Debugging flags
$DEBUG_CXXFLAGS = ' -g';
$DEBUG_CUDACCFLAGS = ' -g';
$DEBUG_LINKFLAGS = ' -lcublas';

# Profiling flags
$PROFILE_CXXFLAGS = ' -O3 -funroll-loops -pg';
$PROFILE_CUDACCFLAGS = ' -O3 --compiler-options="-O3 -funroll-loops -pg"';
$PROFILE_LINKFLAGS = ' -pg -lcublas';

# Disassembly flags
$DISASSEMBLE_CUDACCFLAGS = ' -keep';
$DISASSEMBLE_LINKFLAGS = ' -lcublas';

# Ocelot flags
$OCELOT_LINKFLAGS = '-L/usr/local/ocelot/lib -lOcelotIr -lOcelotParser -lOcelotExecutive -lOcelotTrace -lOcelotAnalysis -lhydrazine -lcublas';

# Device emulation flags
$EMULATION_CUDACCFLAGS .= ' --device-emulation -g';
$EMULATION_LINKFLAGS = ' --device-emulation -g -lcublasemu';

# Bootstrap
`mkdir -p $BUILDDIR $CPPDIR`;
`$SPEC2XDIR/csv2sql --model $NAME --outdir $BUILDDIR --srcdir $SPEC2XSRCDIR < $SPECDIR/$SPEC.csv`;
`$SPEC2XDIR/sql2cpp --outdir $CPPDIR --ttdir $SPEC2XCPPTTDIR --model $BUILDDIR/$NAME.db`;

# Walk through source
@files = ($SRCDIR);
while (@files) {
  $file = shift @files;
  if (-d $file) {
    # recurse into directory
    opendir(DIR, $file);
    push(@files, map { "$file/$_" } grep { !/^\./ } readdir(DIR));
    closedir(DIR);
  } elsif (-f $file && $file =~ /\.(cu|c|cpp)$/) {
    $ext = $1;

    # target name
    $target = $file;
    $target =~ s/^$SRCDIR/$BUILDDIR/;
    $target =~ s/\.\w+$/.$ext.o/;

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
    $target =~ /(.*)\//;
    $dir = $1;
    $dirs{$dir} = 1;

    $command = `$cc $flags -M $file`;
    $command =~ s/.*?\:\w*//;
    $command = "$target: " . $command;
    $command .= "\tmkdir -p $dir\n";
    $command .= "\t$ccstr -o $target $flagstr -c $file\n";
    $command .= "\trm -f *.linkinfo\n";
    push(@targets, $target);
    push(@commands, $command);
    if ($dir eq "$BUILDDIR/model") {
      push(@models, $target);
    }
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

DEBUG_CXXFLAGS=$DEBUG_CXXFLAGS
DEBUG_CUDACCFLAGS=$DEBUG_CUDACCFLAGS
DEBUG_LINKFLAGS=$DEBUG_LINKFLAGS

RELEASE_CXXFLAGS=$RELEASE_CXXFLAGS
RELEASE_CUDACCFLAGS=$RELEASE_CUDACCFLAGS
RELEASE_LINKFLAGS=$RELEASE_LINKFLAGS

PROFILE_CXXFLAGS=$PROFILE_CXXFLAGS
PROFILE_CUDACCFLAGS=$PROFILE_CUDACCFLAGS
PROFILE_LINKFLAGS=$PROFILE_LINKFLAGS

DISASSEMBLE_CXXFLAGS=$DISASSEMBLE_CXXFLAGS
DISASSEMBLE_CUDACCFLAGS=$DISASSEMBLE_CUDACCFLAGS
DISASSEMBLE_LINKFLAGS=$DISASSEMBLE_LINKFLAGS

OCELOT_CXXFLAGS=$OCELOT_CXXFLAGS
OCELOT_CUDACCFLAGS=$OCELOT_CUDACCFLAGS
OCELOT_LINKFLAGS=$OCELOT_LINKFLAGS

EMULATION_CXXFLAGS=$EMULATION_CXXFLAGS
EMULATION_CUDACCFLAGS=$EMULATION_CUDACCFLAGS
EMULATION_LINKFLAGS=$EMULATION_LINKFLAGS

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

ifdef USE_TEXTURE
CUDACCFLAGS += -DUSE_TEXTURE
CXXFLAGS += -DUSE_TEXTURE
endif

ifdef NDEBUG
CUDACCFLAGS += -DNDEBUG
CXXFLAGS += -DNDEBUG
endif

ifdef DEBUG
CUDACCFLAGS += \$(DEBUG_CUDACCFLAGS)
CXXFLAGS += \$(DEBUG_CXXFLAGS)
LINKFLAGS += \$(DEBUG_LINKFLAGS)
endif

ifdef RELEASE
CUDACCFLAGS += \$(RELEASE_CUDACCFLAGS)
CXXFLAGS += \$(RELEASE_CXXFLAGS)
LINKFLAGS += \$(RELEASE_LINKFLAGS)
endif

ifdef PROFILE
CUDACCFLAGS += \$(PROFILE_CUDACCFLAGS)
CXXFLAGS += \$(PROFILE_CXXFLAGS)
LINKFLAGS += \$(PROFILE_LINKFLAGS)
endif

ifdef DISASSEMBLE
CUDACCFLAGS += \$(DISASSEMBLE_CUDACCFLAGS)
CXXFLAGS += \$(DISASSEMBLE_CXXFLAGS)
LINKFLAGS += \$(DISASSEMBLE_LINKFLAGS)
endif

ifdef OCELOT
CUDACCFLAGS += \$(OCELOT_CUDACCFLAGS)
CXXFLAGS += \$(OCELOT_CXXFLAGS)
LINKFLAGS += \$(OCELOT_LINKFLAGS)
endif

ifdef EMULATION
CUDACCFLAGS += \$(EMULATION_CUDACCFLAGS)
CXXFLAGS += \$(EMULATION_CXXFLAGS)
LINKFLAGS += \$(EMULATION_LINKFLAGS)
endif

End

# Default targets
print <<End;
default: \$(BUILDDIR)/simulate \$(BUILDDIR)/filter \$(BUILDDIR)/mcmc \$(BUILDDIR)/ukf

End

# spec2x targets
print <<End;

\$(BUILDDIR)/\$(NAME).db: \$(SPECDIR)/\$(SPEC).csv \$(SPEC2XDIR)/csv2sql \$(SPEC2XSRCDIR)/sqlite.sql
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XDIR)/csv2sql --model \$(NAME) --outdir \$(BUILDDIR) --srcdir \$(SPEC2XSRCDIR) < \$(SPECDIR)/\$(SPEC).csv 

\$(BUILDDIR)/\$(NAME).dot: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XDIR)/sql2dot
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XDIR)/sql2dot --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db > \$(BUILDDIR)/\$(NAME).dot

\$(BUILDDIR)/\$(NAME)_graph.tex: \$(BUILDDIR)/\$(NAME).dot
\tmkdir -p \$(BUILDDIR)
\tdot2tex --autosize --usepdflatex --figpreamble="\huge" --prog=neato -traw --crop < \$(BUILDDIR)/\$(NAME).dot > \$(BUILDDIR)/\$(NAME)_graph.tex

\$(BUILDDIR)/\$(NAME)_graph.pdf: \$(BUILDDIR)/\$(NAME)_graph.tex
\tmkdir -p \$(BUILDDIR)
\tcd \$(BUILDDIR); \\
\tpdflatex \$(NAME)_graph.tex; \\
\tpdflatex \$(NAME)_graph.tex; \\
\tcd ..

\$(BUILDDIR)/\$(NAME).tex: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XDIR)/sql2tex \$(SPEC2XTEXTEMPLATEDIR)/*.template
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XDIR)/sql2tex --templatedir \$(SPEC2XTEXTEMPLATEDIR) --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db > \$(BUILDDIR)/\$(NAME).tex

\$(BUILDDIR)/\$(NAME).pdf: \$(BUILDDIR)/\$(NAME).tex
\tmkdir -p \$(BUILDDIR)
\tcd \$(BUILDDIR); \\
\tpdflatex \$(NAME).tex; \\
\tpdflatex \$(NAME).tex; \\
\tcd ..

sql: \$(BUILDDIR)/\$(NAME).db

dot: \$(BUILDDIR)/\$(NAME)_graph.pdf

cpp: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XDIR)/sql2cpp \$(SPEC2XCPPTTDIR)/*.tt
\tmkdir -p \$(CPPDIR)
\tperl \$(SPEC2XDIR)/sql2cpp --outdir \$(CPPDIR) --ttdir \$(SPEC2XCPPTTDIR) --model \$(BUILDDIR)/\$(NAME).db

tex: \$(BUILDDIR)/\$(NAME).tex

pdf: \$(BUILDDIR)/\$(NAME).pdf

End

# Artefacts
my $models = join(' ', @models);

print "\$(BUILDDIR)/simulate: \$(BUILDDIR)/simulate.cpp.o \$(BUILDDIR)/simulate.cu.o $models\n";
print "\t\$(LINKER) -o $BUILDDIR/simulate \$(BUILDDIR)/simulate.cpp.o \$(BUILDDIR)/simulate.cu.o $models \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/filter: \$(BUILDDIR)/filter.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o $models\n";
print "\t\$(LINKER) -o $BUILDDIR/filter \$(BUILDDIR)/filter.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o $models \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/ukf: \$(BUILDDIR)/ukf.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o $models\n";
print "\t\$(LINKER) -o $BUILDDIR/ukf \$(BUILDDIR)/ukf.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o $models \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/mcmc: \$(BUILDDIR)/mcmc.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o \$(BUILDDIR)/device.cu.o $models\n";
print "\t\$(LINKER) -o $BUILDDIR/mcmc \$(BUILDDIR)/mcmc.cpp.o \$(BUILDDIR)/filter.cu.o \$(BUILDDIR)/prior.cpp.o \$(BUILDDIR)/device.cu.o $models \$(LINKFLAGS)\n\n";

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

