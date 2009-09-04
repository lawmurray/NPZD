##
## Reads a CSV model specification and populates an SQLite database.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

use strict;
use Text::CSV;
use DBI;
use DBD::SQLite;
use Pod::Usage;
use Getopt::Long;
use List::MoreUtils qw/uniq/;

# command line arguments
my $model = 'MyModel';
my $outdir = '.';
my $srcdir = 'src';
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "outdir=s" => \$outdir,
    "srcdir=s" => \$srcdir);

# create sqlite database
my $dbfile = "$outdir/$model.db";
`sqlite3 $dbfile < $srcdir/sqlite.sql`;

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$dbfile", '', '', { AutoCommit => 0 });

# database statement handles
my %sth;
$sth{'InsertNode'} = $dbh->prepare('INSERT INTO Node(Name,LaTeXName,' .
    'Description,Category) VALUES (?,?,?,?)');
$sth{'InsertNodeTrait'} = $dbh->prepare('INSERT INTO NodeTrait(Node,' .
    'Trait) VALUES (?,?)');
$sth{'InsertNodeFormula'} = $dbh->prepare('INSERT INTO NodeFormula(Node,' .
    'Function,Formula,LaTeXFormula) VALUES (?,?,?,?)');
$sth{'InsertEdge'} = $dbh->prepare('INSERT INTO Edge(ParentNode,ChildNode,' .
    'Position) VALUES (?,?,?)');
$sth{'UpdatePosition'} = $dbh->prepare('UPDATE Node SET Position = ? WHERE ' .
    'Name = ?');

# process CSV headers
my $io = \*STDIN;
my $csv = Text::CSV->new();
$csv->column_names($csv->getline($io));

# process CSV
my $fields;
my $val;
my $pos;
my @nodes;
my @parents;
my @children;
my @positions;
my %dependents;
my $i;
my %formulae;
my %latexformulae;
my $function;
my $formula;

$fields = $csv->getline_hr($io);
while (!$csv->eof()) {
  push(@nodes, $$fields{'Name'});

  # insert node
  $sth{'InsertNode'}->execute($$fields{'Name'}, $$fields{'LaTeXName'},
      $$fields{'Description'}, $$fields{'Category'}) ||
      warn("Problem with node $$fields{'Name'}");

  # insert traits
  foreach $val (split /,\s*/, $$fields{'Traits'}) {
    $sth{'InsertNodeTrait'}->execute($$fields{'Name'}, uc($val)) ||
        warn("Problem with trait $val of node $$fields{'Name'}");
  }

  # insert formulas
  undef %formulae;
  undef %latexformulae;
  foreach $val (split /\s*\|\s*/, $$fields{'Formulae'}) {
    ($function, $formula) = split(/\s*:\s*/, $val);
    $formulae{$function} = $formula;
  }
  foreach $val (split /\s*\|\s*/, $$fields{'LaTeXFormulae'}) {
    ($function, $formula) = split(/\s*:\s*/, $val);
    $latexformulae{$function} = $formula;
  }
  foreach $function (uniq(keys %formulae, keys %latexformulae)) {
    $sth{'InsertNodeFormula'}->execute($$fields{'Name'}, $function,
        $formulae{$function}, $latexformulae{$function});
  }

  # store edges for later
  $pos = 0;
  %{$dependents{$$fields{'Name'}}} = ();
  foreach $val (split /,\s*/, $$fields{'Dependencies'}) {
    if ($$fields{'Traits'} =~ /\bIS_S_NODE\b/) {
      # need to topologically order
      $dependents{$$fields{'Name'}}{$val} = 1;
    }
    push(@parents, $val);
    push(@children, $$fields{'Name'});
    push(@positions, $pos);
    ++$pos;
  }

  # next
  $fields = $csv->getline_hr($io);
}

# update positions based on topological sort
my @sorted = &TopologicalSort(\@nodes, \%dependents);
for ($i = 0; $i < @sorted; ++$i) {
  $sth{'UpdatePosition'}->execute($i, $sorted[$i]);
}

# insert dependencies as edges
for ($i = 0; $i < @parents; ++$i) {
  $sth{'InsertEdge'}->execute($parents[$i], $children[$i], $positions[$i]) ||
       warn("Problem with dependency $parents[$i] of node $children[$i]\n");
}

# wrap up
my $key;
foreach $key (keys %sth) {
  $sth{$key}->finish;
}
$dbh->commit;
undef %sth; # workaround for SQLite warning about active statement handles
$dbh->disconnect;

##
## Topologically sort s-nodes so that no node appears before all of its parents
## have appeared.
##
## @param dependencies Hash-of-hashes, outside keyed by s-node, inside
## list of parents of that node (of any types). Destroyed in process.
## 
## @return Sorted array of node names.
##
sub TopologicalSort {
  my $nodes = shift;
  my $dependencies = shift;
  my @result;
  my $key;
  my $node;
  my $i;

  # flush self-dependencies and non s-node dependencies
  foreach $key (keys %$dependencies) {
    delete $dependencies->{$key}{$key};
    foreach $node (keys %{$dependencies->{$key}}) {
      if (!exists $dependencies->{$node}) {
        delete $dependencies->{$key}{$node};
      }
    }
  }
  
  # sort
  while (@$nodes) {
    # find node with all dependencies satisfied
    $i = 0;
    while ($i < @$nodes && keys %{$dependencies->{$$nodes[$i]}} > 0) {
      ++$i;
    }
    if ($i >= @$nodes) {
      $i = 0;
      warn('S-nodes have no partial order, loop exists?');
    }

    $node = $$nodes[$i];
    splice(@$nodes, $i, 1);

    push(@result, $node);
    delete $dependencies->{$node};
    
    # delete this node from dependency lists
    foreach $key (keys %$dependencies) {
      delete $dependencies->{$key}{$node};
    }
  }
  
  return @result;
}

__END__

=head1 NAME

csv2sql -- CSV to SQLite database converter for model specifications.

=head1 SYNOPSIS

csv2sql [options]

The CSV model specification is read from stdin. The SQLite database will be
written to $model.db, where $model is specified as a command line argument.

=head1 

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--model>

Specify the model name.

=item B<--output>

Specify the output directory.

=item B<--srcdir>

Specify the source directory (the directory in which the sqlite.sql script
resides).

=back

=head1 DESCRIPTION

Reads a CSV model specification from stdin and generates an SQLite database
representation of the same.

=cut

