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
use DBD::SQLite; # includes SQLite, not needed separately
use Pod::Usage;
use Getopt::Long;

# command line arguments
my $model = 'MyModel';
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model);

# create sqlite database
my $dbfile = "$model.db";
`sqlite3 $dbfile < src/sqlite.sql`;

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

# process CSV headers
my $io = \*STDIN;
my $csv = Text::CSV->new();
$csv->column_names($csv->getline($io));

# process CSV
my $fields;
my $val;
my $pos;
my @parents;
my @children;
my @positions;

$fields = $csv->getline_hr($io);
while (!$csv->eof()) {
  # Insertnode
  $sth{'InsertNode'}->execute($$fields{'Name'}, $$fields{'LaTeXName'},
      $$fields{'Formula'}, $$fields{'LaTeXFormula'}, $$fields{'Description'},
      $$fields{'Type'}) ||
      warn("Problem with node $$fields{'Name'}");

  # Inserttraits
  foreach $val (split /,\s*/, $$fields{'Traits'}) {
    $sth{'InsertNodeTrait'}->execute($$fields{'Name'}, uc($val)) ||
        warn("Problem with trait $val of node $$fields{'Name'}");
  }

  # store edges for later
  $pos = 0;
  foreach $val (split /,\s*/, $$fields{'Dependencies'}) {
    push(@parents, $val);
    push(@children, $$fields{'Name'});
    push(@positions, $pos);
    ++$pos;
  }

  # next
  $fields = $csv->getline_hr($io);
}

# Insertdependencies as edges
my $i;
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

=back

=head1 DESCRIPTION

Reads a CSV model specification from stdin and generates an SQLite database
representation of the same.

=cut

