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
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model);

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$model.db", '', '',
    { AutoCommit => 0 });

# database statement handles
my %sth;
$sth{'GetNodes'} = $dbh->prepare('SELECT Name, LaTeXName, Formula, ' .
    'LaTeXFormula, Type, Description FROM Node');
$sth{'GetEdges'} = $dbh->prepare('SELECT ParentNode, ChildNode, Type FROM ' .
    'Edge, Node WHERE Edge.ChildNode = Node.Name ORDER BY Position');
$sth{'CheckTrait'} = $dbh->prepare('SELECT 1 FROM NodeTrait WHERE ' .
    'Node = ? AND Trait = ?');

# output header
print <<End;
digraph model {
  overlap=scale;
  splines=true;
  sep=.4;
  d2tgraphstyle="scale=0.8"
  nslimit=8.0;
  mclimit=8.0;
End
#  ratio=0.71;  # for A4 landscape

my $fields;
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
  $style = &nodeStyle($$fields{'Type'});
  print qq/  $$fields{'Name'} \[texlbl="$label",$style\]\n/;

  # description label node
  $label = '';
  if ($$fields{'Description'} ne '') {
    $str = $$fields{'Description'};
    $str =~ s/\n/\\\\/g;
    $label .= "$str";
    $label .= "\\\\";
  }

  if ($$fields{'LaTeXFormula'} ne '') {
    $sth{'CheckTrait'}->execute($$fields{'Name'}, 'IS_ODE_FORWARD');
    if ($sth{'CheckTrait'}->fetchrow_array) {
      $formula = "\\dot\{$$fields{'LaTeXName'}\} = $$fields{'LaTeXFormula'}";
    } else {
      $formula = $$fields{'LaTeXFormula'};
    }
    $label .= &mathLabel(&escapeLabel($formula));
  } elsif ($$fields{'Formula'} ne '') {
    $label .= $$fields{'Formula'};
  }

  if ($label ne '') {
    $style = labelEdgeStyle($$fields{'Type'});
    print qq/  $$fields{'Name'}\_label \[texlbl="\\parbox{5cm}{$label}",shape=plaintext\]\n/;
    print qq/  $$fields{'Name'}\_label -> $$fields{'Name'} \[arrowhead=none,len=.1,$style\]\n/;
  }
}

# output edges
$sth{'GetEdges'}->execute;
while ($fields = $sth{'GetEdges'}->fetchrow_hashref) {
  if ($$fields{'Type'} eq 'Dynamic parameter' &&
      $$fields{'ParentNode'} =~ /^(?:alpha|sigma|u)/) {
    print qq/  $$fields{'ParentNode'} -> $$fields{'ChildNode'} [len=.1];\n/;
  } else {
    print "  $$fields{'ParentNode'} -> $$fields{'ChildNode'} [len=.3];\n";
  }
}

# output legend
my $name;
my $type;
my @types = ('Constant', 'Forcing', 'Random variate', 'Dynamic parameter', 'Static parameter', 'State variable', 'Intermediate result');
print qq/  subgraph legend {\n/;
print qq/    label="Legend"\n/;
foreach $type (@types) {
  $style = nodeStyle($type);
  $name = $type;
  $name =~ s/\s/_/g;
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
    'State variable' => 'circle',
    'Constant' => 'box',
    'Forcing' => 'box',
    'Dynamic parameter' => 'circle',
    'Static parameter' => 'circle',
    'Intermediate result' => 'circle',
    'Random variate' => 'circle'
  );  
  my %STYLES = (
    'State variable' => 'filled',
    'Constant' => 'filled',
    'Forcing' => 'filled',
    'Dynamic parameter' => 'filled',
    'Static parameter' => 'filled',
    'Intermediate result' => 'dashed',
    'Random variate' => 'filled'
  );
  my %COLORS = (
    'State variable' => '#6677FF',
    'Constant' => '#FFCC33',
    'Forcing' => '#FF6666',
    'Dynamic parameter' => '#66EE77',
    'Static parameter' => '#66EE77',
    'Intermediate result' => '#000000',
    'Random variate' => '#CC79A7'
  );
  my %FILLCOLORS = (
    'State variable' => '#BBCCFF',
    'Constant' => '#FFEEAA',
    'Forcing' => '#FFBBBB',
    'Dynamic parameter' => '#BBEECC',
    'Static parameter' => '#FFFFFF',
    'Intermediate result' => '#FFFFFF',
    'Random variate' => '#FFA9D7'
  );
  my %FONTCOLORS = (
    'State variable' => '#000000',
    'Constant' => '#000000',
    'Forcing' => '#000000',
    'Dynamic parameter' => '#000000',
    'Static parameter' => '#000000',
    'Intermediate result' => '#000000',
    'Random variate' => '#000000'
  );

  my $style = qq/shape="$SHAPES{$type}",style="$STYLES{$type}",color="$COLORS{$type}",fillcolor="$FILLCOLORS{$type}",fontcolor="$FONTCOLORS{$type}"/;

  return $style;
}

##
## Construct style string for edge type.
##
sub labelEdgeStyle {
  my $type = shift;
  my %STYLES = (
    'State variable' => 'solid',
    'Constant' => 'solid',
    'Forcing' => 'solid',
    'Dynamic parameter' => 'solid',
    'Static parameter' => 'solid',
    'Intermediate result' => 'dashed'
  );
  my %COLORS = (
    'State variable' => '#6677FF',
    'Constant' => '#FFCC33',
    'Forcing' => '#FF6666',
    'Dynamic parameter' => '#66EE77',
    'Static parameter' => '#66EE77',
    'Intermediate result' => '#000000'
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

=back

=head1 DESCRIPTION

Reads an SQLite database model specification and generates a PDF graphical
visualisation.

=cut

