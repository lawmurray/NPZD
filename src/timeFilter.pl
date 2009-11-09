#!/usr/bin/perl

$K = 8; # size of intermediate result buffer
$T = 365.0; # time to simulate
$H = 0.3; # initial step size
$SEED = 0; # pseudorandom number seed
$OUTPUT = 0; # produce output?
$TIME = 1; # produce timings?

$INIT_FILE = "data/GPUinput_OSP_0D.nc"; # initial values file
$FORCE_FILE = "data/GPUinput_OSP_0D.nc"; # forcings file
$OBS_FILE = "data/GPUobs_EQ.nc"; # observations file
$OUTPUT_FILE = "npzd/results/pf.nc"; # output file

$INIT_NS = 0;
$FORCE_NS = 0;
$OBS_NS = 1;

# experiments
my %exps;
$exps{'single'} = 'RELEASE=1 NDEBUG=1';
$exps{'double'} = 'RELEASE=1 USE_DOUBLE=1 NDEBUG=1';
$exps{'texture'} = 'RELEASE=1 USE_TEXTURE=1 NDEBUG=1';

foreach $exp (keys %exps) {
  # compile library
  chdir('../bi');
  `make clean`;
  `make -j $exps{$exp}`;
  
  # compile experiment
  chdir('../npzd');
  `make clean`;
  `make -j $exps{$exp} build/filter`;
  
  open(FILE, ">results/timeFilter_$exp.csv") || die;
  for ($i = 1; $i <= 10; ++$i) {
    $P = $i*1024;
    print "P=$P...\n";
    print FILE "$P\t";
    print FILE `build/filter -P $P -K $K -T $T -h $H --seed $SEED --init-file $INIT_FILE --force-file $FORCE_FILE --obs-file $OBS_FILE --init-ns $INIT_NS --force-ns $FORCE_NS --obs-ns $OBS_NS --output-file $OUTPUT_FILE --output $OUTPUT --time $TIME`;
  }
  close FILE;
}

