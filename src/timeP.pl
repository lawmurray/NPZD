#!/usr/bin/perl

$K = 365;
$T = 365.0;
$SEED = 0;
$OUTPUT="--output 0 --time 1";

# experiments
my %exps;
$exps{'DOPRI5_single'} = 'USE_DOPRI5=1 NDEBUG=1';
$exps{'DOPRI5_double'} = 'USE_DOUBLE=1 USE_DOPRI5=1 NDEBUG=1';
$exps{'DOPRI5_intrinsic'} = 'USE_DOPRI5=1 USE_FAST_MATH=1 NDEBUG=1';
$exps{'RK43_single'} = 'NDEBUG=1';
$exps{'RK43_double'} = 'USE_DOUBLE=1 NDEBUG=1';
$exps{'RK43_intrinsic'} = 'USE_FAST_MATH=1 NDEBUG=1';

foreach $exp (keys %exps) {
  # compile library
  chdir('../bi');
  `make clean`;
  `make $exps{$exp}`;
  
  # compile experiment
  chdir('../npzd');
  `make clean`;
  `make $exps{$exp}`;
  
  open(FILE, ">results/timeP_$exp.csv") || die;
  $P = 1;
  for ($i = 0; $i < 17; ++$i) {
    print FILE $P;
    print FILE `build/simulate --seed $SEED $OUTPUT -P $P -K $K -T $T`;
    print FILE "\n";
    $P *= 2;
  }
  close FILE;
}
