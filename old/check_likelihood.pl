#!/usr/bin/perl

##
## Checks that all jobs have completed successfully, and generates qsub
## commands to rerun those that have not.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$ $Date$
##

#chdir '/home/mur387/sandbox_pz/pz';
$RESULTS_DIR = 'results';
@EXPS = ('pf', 'apf', 'mupf', 'amupf', 'cupf', 'acupf', 'pf-pmatch', 'apf-pmatch', 'mupf-pmatch', 'amupf-pmatch');
$M = 16;

foreach $exp (@EXPS) {
    @t = ();
    for ($m = 0; $m < $M; ++$m) {
	@files = <$RESULTS_DIR/likelihood_$exp-$m.nc.*>;

	# should be four files per job
	$redo = @files != 4;

	# all file sizes should be the same
	$redo ||= !same_size(\@files);

	# add this array job id if need to redo
	if ($redo) {
	    push(@t, $m);
	}
    }

    print "$exp redoing " . scalar(@t) . "\n";
    if (@t > 0) {
	$t = join(',', @t);
	print `qsub -N $exp -t $t likelihood.sh`;
    }
}

sub same_size {
    my $files = shift;
    my $i;
    my $same = 0;

    for ($i = 0; $i < @$files; ++$i) {
	if ($i == 0) {
	    $size = -s $$files[$i];
	} else {
	    $same = $size == -s $$files[$i];
	}
    }

    return $same;
}
