##
## Generate a Makefile for compilation.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

$CXX = 'g++';
$CUDACC = 'nvcc';
$CXXFLAGS = '-Wall -g -O3 -I"../bi/src" `nc-config --cflags`';
$CUDACCFLAGS = '-O3 -g -arch=sm_13 -Xptxas="-v" -I"../bi/src" -DBOOST_NO_INCLASS_MEMBER_INITIALIZATION -DBOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS `nc-config --cflags`';
$LINKFLAGS = '-L"../bi/build" `nc-config --libs` -lnetcdf_c++ -lboost_program_options-gcc41-mt -lblas -llapack -lgfortran -lgslcblas -lgsl -lbi';
# ^ may need f2c, g2c or nothing in place of gfortran
$DEPFLAGS = '-I"../bi/src"'; # flags for dependencies check

$NAME = 'NPZD';
$SRCDIR = 'src';
$BUILDDIR = 'build';
$SPECDIR = 'spec';
$SPEC2XDIR = 'spec2x';
$SPEC2XSRCDIR = "$SPEC2XDIR/src";
$SPEC2XTEMPLATEDIR = "$SPEC2XDIR/templates";
$CPPDIR = "$SRCDIR/model";

# Disassembly
#$CXXFLAGS .= ' -keep';

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

    $command = "$dir/" . `$cc $DEPFLAGS -M $file`;
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

BUILDDIR=$BUILDDIR
SRCDIR=$SRCDIR
SPECDIR=$SPECDIR
SPEC2XDIR=$SPEC2XDIR
SPEC2XSRCDIR=$SPEC2XSRCDIR
SPEC2XTEMPLATEDIR=$SPEC2XTEMPLATEDIR
CPPDIR=$CPPDIR

CXX=$CXX
CXXFLAGS=$CXXFLAGS
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

# Default target
print <<End;
default: \$(BUILDDIR)/simulate

End

# spec2x targets
print <<End;

\$(BUILDDIR)/\$(NAME).db: \$(SPECDIR)/\$(NAME).csv \$(SPEC2XSRCDIR)/csv2sql.pl \$(SPEC2XSRCDIR)/sqlite.sql
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XSRCDIR)/csv2sql.pl --model \$(NAME) --outdir \$(BUILDDIR) --srcdir \$(SPEC2XSRCDIR) < \$(SPECDIR)/\$(NAME).csv 

\$(BUILDDIR)/\$(NAME).dot: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XSRCDIR)/sql2dot.pl
\tmkdir -p \$(BUILDDIR)
\tperl \$(SPEC2XSRCDIR)/sql2dot.pl --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db > \$(BUILDDIR)/\$(NAME).dot

\$(BUILDDIR)/\$(NAME).tex: \$(BUILDDIR)/\$(NAME).dot
\tmkdir -p \$(BUILDDIR)
\tdot2tex --autosize --usepdflatex --figpreamble="\huge" --prog=neato -traw --crop < \$(BUILDDIR)/\$(NAME).dot > \$(BUILDDIR)/\$(NAME).tex

\$(BUILDDIR)/\$(NAME).pdf: \$(BUILDDIR)/\$(NAME).tex
\tmkdir -p \$(BUILDDIR)
\tcd \$(BUILDDIR); \\
\tpdflatex \$(NAME).tex; \\
\tpdflatex \$(NAME).tex; \\
\tcd ..

sql: \$(BUILDDIR)/\$(NAME).db

dot: \$(BUILDDIR)/\$(NAME).pdf

cpp: \$(BUILDDIR)/\$(NAME).db \$(SPEC2XSRCDIR)/sql2cpp.pl \$(SPEC2XTEMPLATEDIR)/*.template
\tmkdir -p \$(CPPDIR)
\tperl \$(SPEC2XSRCDIR)/sql2cpp.pl --outdir \$(CPPDIR) --templatedir \$(SPEC2XTEMPLATEDIR) --model \$(NAME) --dbfile \$(BUILDDIR)/\$(NAME).db

End

# Model target
print <<End;
\$(BUILDDIR)/\$(NAME).o: cpp
\t\$(CUDACC) -c -o \$(BUILDDIR)/\$(NAME).o \$(CUDACCFLAGS) \$(CPPDIR)/\$(NAME).cu

End

# Artefact
print "\$(BUILDDIR)/simulate: cpp ";
print join(" \\\n    ", @targets);
print "\n";
#print "\tar -r \$(BUILDDIR)/simulate \$(LINKFLAGS) " . join(' ', @targets);
print "\t$CUDACC -o $BUILDDIR/simulate \$(LINKFLAGS) " . join(' ', @targets);

print "\n\n";

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
