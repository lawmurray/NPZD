#!/usr/bin/perl

for ($chains = 4; $chains <= 16; $chains *= 2) {
    gen_script('dmcmc-share', $chains);
    gen_script('dmcmc-noshare', $chains);
    gen_script('haario', $chains);
    gen_script('straight', $chains);

    qsub_script('dmcmc-share', $chains);
    qsub_script('dmcmc-noshare', $chains);
    qsub_script('haario', $chains);
    qsub_script('straight', $chains);
}

##
## Submit script.
##
sub qsub_script {
    my $name = shift;
    my $chains = shift;
    my $script = "converge-across-$name-$chains.sh";

    print `qsub $script`; 
}

##
## Generate script for submission.
##
sub gen_script {
    my $name = shift;
    my $chains = shift;
    my $script = "converge-across-$name-$chains.sh";

    open(SCRIPT, ">$script") || die("Could not open $script for writing");
print SCRIPT <<End;
#!/bin/bash
#PBS -l walltime=04:00:00,nodes=1:ppn=4,vmem=16gb
#PBS -j oe

module load octave/3.3.51-gnu

ROOT=\$PBS_O_WORKDIR
NAME=$name
C=$chains
A=40 # number of ensembles (numbered 0 to A - 1 as array job)
S=25000 # number of steps in each chain

cd \$ROOT

octave src/converge_across.m \$NAME \$A \$C \$S
End
    close SCRIPT;
}
