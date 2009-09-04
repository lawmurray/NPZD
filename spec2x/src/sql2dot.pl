##
## Convert SQLite model specification to dot script.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

use strict;
use DBI;
use DBD::SQLite;
use Pod::Usage;
use Getopt::Long;

# command line arguments
my $model = 'MyModel';
my $dbfile = "$model.db";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "dbfile=s" => \$dbfile);

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$dbfile", '', '',
    { AutoCommit => 0 });

# database statement handles
my %sth;
$sth{'GetNodes'} = $dbh->prepare('SELECT Name, LaTeXName, ' .
    'Category, Description FROM Node');
$sth{'GetEdges'} = $dbh->prepare('SELECT ParentNode, ChildNode, Category ' .
    'FROM Edge, Node WHERE Edge.ParentNode = Node.Name ORDER BY Edge.Position');
$sth{'CheckTrait'} = $dbh->prepare('SELECT 1 FROM NodeTrait WHERE ' .
    'Node = ? AND Trait = ?');
$sth{'GetNodeFormulae'} = $dbh->prepare('SELECT Function, LaTeXFormula ' .
    'FROM NodeFormula WHERE Node = ?');

# output header
print <<End;
digraph model {
  overlap=scale;
  splines=true;
  sep=.4;
  d2tgraphstyle="scale=0.8"
  nslimit=2.0;
  mclimit=2.0;
  ratio=0.71;  # for A4 landscape
End

my $fields;
my $formulafields;
my $label;
my $style;
my $str;
my $formula;

# output nodes
$sth{'GetNodes'}->execute;
while ($fields = $sth{'GetNodes'}->fetchrow_hashref) {
  # variable node
  if ($$fields{'LaTeXName'} ne '') {
    $label = &mathLabel(&escapeLabel($$fields{'LaTeXName'}));
  } else {
    $label = $$fields{'Name'};
  }
  $style = &nodeStyle($$fields{'Category'});
  print qq/  $$fields{'Name'} \[texlbl="$label",$style\]\n/;

  # description label node
  $label = '';
  if ($$fields{'Description'} ne '') {
    $str = $$fields{'Description'};
    $str =~ s/\n/\\\\/g;
    $label .= "$str";
    $label .= "\\\\";
  }

  # ...add formulae
  $sth{'GetNodeFormulae'}->execute($$fields{'Name'});
  while ($formulafields = $sth{'GetNodeFormulae'}->fetchrow_hashref) {
    if ($$formulafields{'LaTeXFormula'} ne '') {
      if ($$formulafields{'Function'} eq 'dfdt') {
        $formula = "\\dot\{$$fields{'LaTeXName'}\} = " .
            $$formulafields{'LaTeXFormula'};
      } elsif ($$formulafields{'Function'} eq 'f') {
        $formula = $$formulafields{'LaTeXFormula'};
      } else {
        $formula = "\\dot\{$$formulafields{'Function'}\} = " .
            $$formulafields{'LaTeXFormula'};
      }
      $label .= &mathLabel(&escapeLabel($formula));
    } elsif ($$formulafields{'Formula'} ne '') {
      $label .= "$$formulafields{'Function'} = $$formulafields{'Formula'}";
    }
  }

  # special case formulae
  $sth{'CheckTrait'}->execute($$fields{'Name'}, 'IS_GAUSSIAN');
  if ($sth{'CheckTrait'}->fetchrow_array) {
    $label .= &mathLabel("\\mathcal{N}(0,1)");
  }
  $sth{'CheckTrait'}->finish;

  if ($label ne '') {
    $style = labelEdgeStyle($$fields{'Category'});
    print qq/  $$fields{'Name'}\_label \[texlbl="\\parbox{5cm}{$label}",shape=plaintext\]\n/;
    print qq/  $$fields{'Name'}\_label -> $$fields{'Name'} \[arrowhead=none,len=.1,$style\]\n/;
  }
}

# output edges
$sth{'GetEdges'}->execute;
while ($fields = $sth{'GetEdges'}->fetchrow_hashref) {
  if ($$fields{'Category'} eq 'Static variable' || $$fields{'Category'} eq 'Random variate') {
    print qq/  $$fields{'ParentNode'} -> $$fields{'ChildNode'} [len=.1];\n/;
  } else {
    print "  $$fields{'ParentNode'} -> $$fields{'ChildNode'} [len=.3];\n";
  }
}

# output legend
my $name;
my $type;
my @types = ('Constant', 'Intermediate result', 'Forcing', 'Observation', 'Random variate', 'Static variable', 'Discrete-time variable', 'Continuous-time variable');
print qq/  subgraph legend {\n/;
print qq/    label="Legend"\n/;
foreach $type (@types) {
  $style = nodeStyle($type);
  $name = &safeName($type);
  $label = substr($type, 0, 1);
  print qq/    legend_node_$name \[label="$label",shape=circle,$style\]\n/;
  print qq/    legend_label_$name \[label="$type",shape=plaintext\]\n/;
  $style = labelEdgeStyle($type);
  print qq/    legend_label_$name -> legend_node_$name \[arrowhead="none",$style\]\n/;
}
my $i;
my $j;
my $name1;
my $name2;
for ($i = 0; $i < @types; ++$i) {
  $name1 = &safeName($types[$i]);
  $name2 = &safeName($types[($i+1) % scalar(@types)]);
  print qq/    legend_node_$name1 -> legend_node_$name2 \[arrowhead="none",style="dotted"]\n/;
}
print "  }\n";

# output footer
print "}\n";

# wrap up
my $key;
foreach $key (keys %sth) {
  $sth{$key}->finish;
}
$dbh->commit;
undef %sth; # workaround for SQLite warning about active statement handles
$dbh->disconnect;

##
## Escape special characters in a label.
##
sub escapeLabel {
  my $str = shift; 
  #$str =~ s/([\\])/\\$1/g;
  return $str;
}

##
## Make name safe as identifier in dot script.
##
sub safeName {
  my $str = shift;
  $str =~ s/\s/_/g;
  $str =~ s/-/_/g;
  return $str;
}

##
## Construct LaTeX math label.
##
sub mathLabel {
  my $str = shift;
  $str = "\$$str\$";
  return $str;
}

##
## Construct style string for node type.
##
sub nodeStyle {
  my $type = shift;
  my %SHAPES = (
    'Constant' => 'box',
    'Random variate' => 'diamond',
    'Forcing' => 'box',
    'Observation' => 'box',
    'Intermediate result' => 'circle',
    'Static variable' => 'circle',
    'Discrete-time variable' => 'circle',
    'Continuous-time variable' => 'circle'
  );  
  my %STYLES = (
    'Constant' => 'dashed',
    'Random variate' => 'filled',
    'Forcing' => 'filled',
    'Observation' => 'filled',
    'Intermediate result' => 'dashed',
    'Static variable' => 'filled',
    'Discrete-time variable' => 'filled',
    'Continuous-time variable' => 'filled'
  );
  my %COLORS = (
    'Constant' => '#000000',
    'Random variate' => '#CC79A7',
    'Forcing' => '#FF6666',
    'Observation' => '#FFCC33',
    'Intermediate result' => '#000000',
    'Static variable' => '#66EE77',
    'Discrete-time variable' => '#66EE77',
    'Continuous-time variable' => '#6677FF'
  );
  my %FILLCOLORS = (
    'Constant' => '#FFFFFF',
    'Random variate' => '#FFA9D7',
    'Forcing' => '#FFBBBB',
    'Observation' => '#FFEEAA',
    'Intermediate result' => '#FFFFFF',
    'Static variable' => '#FFFFFF',
    'Discrete-time variable' => '#BBEECC',
    'Continuous-time variable' => '#BBCCFF'
  );

  my $style = qq/shape="$SHAPES{$type}",style="$STYLES{$type}",color="$COLORS{$type}",fillcolor="$FILLCOLORS{$type}"/;

  return $style;
}

##
## Construct style string for edge type.
##
sub labelEdgeStyle {
  my $type = shift;
  my %STYLES = (
    'Constant' => 'dashed',
    'Random variate' => 'solid',
    'Forcing' => 'solid',
    'Observation' => 'solid',
    'Intermediate result' => 'dashed',
    'Static variable' => 'solid',
    'Discrete-time variable' => 'solid',
    'Continuous-time variable' => 'solid'
  );
  my %COLORS = (
    'Constant' => '#000000',
    'Random variate' => '#CC79A7',
    'Forcing' => '#FF6666',
    'Observation' => '#FFCC33',
    'Intermediate result' => '#000000',
    'Static variable' => '#66EE77',
    'Discrete-time variable' => '#66EE77',
    'Continuous-time variable' => '#6677FF'
  );
  my $style = qq/style="$STYLES{$type}",color="$COLORS{$type}"/;

  return $style;
}

__END__

=head1 NAME

sql2dot -- Graph generation script for model specification.

=head1 SYNOPSIS

sql2dot [options]

The SQLite database is read from $model.db, where $model is specified as a
command line argument. The graph will be written to $model.pdf.

=head1 

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--model>

Specify the model name.

=item B<--dbfile>

Specify the database file name.

=back

=head1 DESCRIPTION

Reads an SQLite database model specification and generates a PDF graphical
visualisation.

=cut

