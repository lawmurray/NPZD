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
    'Formula,LaTeXFormula,Description,Type) VALUES (?,?,?,?,?,?)');
$sth{'InsertNodeTrait'} = $dbh->prepare('INSERT INTO NodeTrait(Node,' .
    'Trait) VALUES (?,?)');
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

$fields = $csv->getline_hr($io);
while (!$csv->eof()) {
  push(@nodes, $$fields{'Name'});

  # insert node
  $sth{'InsertNode'}->execute($$fields{'Name'}, $$fields{'LaTeXName'},
      $$fields{'Formula'}, $$fields{'LaTeXFormula'}, $$fields{'Description'},
      $$fields{'Type'}) ||
      warn("Problem with node $$fields{'Name'}");

  # insert traits
  foreach $val (split /,\s*/, $$fields{'Traits'}) {
    $sth{'InsertNodeTrait'}->execute($$fields{'Name'}, uc($val)) ||
        warn("Problem with trait $val of node $$fields{'Name'}");
  }

  # store edges for later
  $pos = 0;
  %{$dependents{$$fields{'Name'}}} = ();
  foreach $val (split /,\s*/, $$fields{'Dependencies'}) {
    if ($$fields{'Traits'} !~ /\bIS_EX_NODE\b/) {
      # will need to topologically order
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
  $sth{'InsertEdge'}->execute($parents[$i], $children[$i],
       $positions[$i]) ||
       warn("Problem with dependency $parents[$i] of node $children[$i]\n");
}

# wrap up
my $key;
foreach $key (keys %sth) {
  $sth{$key}->finish;
}
$dbh->commit;
$dbh->disconnect;

##
## Topologically sort nodes so that no node appears before all of its parents
## have appeared.
##
## @param dependencies Hash-of-hashes, outside keyed by node, inside
## list of parents of that node. Destroyed in process. Ex-nodes should
## not be included, as they needn't be in topological order,
## although they may be included as dependencies, and will simply be
## ignored.
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

  # flush self-dependencies and non in-node dependencies
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
      die('In-nodes have no partial order, loop exists?');
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

