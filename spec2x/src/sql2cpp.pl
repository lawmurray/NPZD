##
## Convert SQLite model specification to C++ code for bi.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

use strict;
use DBI;
use DBD::SQLite;
use Pod::Usage;
use Carp;
use Getopt::Long;
use List::MoreUtils qw/uniq/;

# command line arguments
my $model = 'MyModel';
my $outdir = '.';
my $templatedir = 'templates';
my $dbfile = "$model.db";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "outdir=s" => \$outdir,
    "templatedir=s" => \$templatedir,
    "dbfile=s" => \$dbfile);

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$dbfile", '', '',
    { AutoCommit => 0 });

# templates
my %templates;
&GulpTemplates;

# output sources
&OutputModelSources;
&OutputNodeSources;

##
## Output source files for model.
##
sub OutputModelSources {
  my $fields;
  my %tokens;
  my @nodes;
  my @classes;
  my %nodes;
  my %classes;
  my $type;
  my $node;
  my $source;

  my $sth = $dbh->prepare('SELECT Name ' .
      "FROM Node WHERE Type <> 'Intermediate result' AND Type <> 'Constant' ".
      'ORDER BY Position ASC');
  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $type = &NodeType($$fields{'Name'});
    push(@nodes, $$fields{'Name'});
    push(@classes, ucfirst($$fields{'Name'}) . 'Node');
    push(@{$nodes{$type}}, $$fields{'Name'});
    push(@{$classes{$type}}, ucfirst($$fields{'Name'}) . 'Node');
  }
  $sth->finish;
  
  # header file
  $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($model) . '_CUH';
  $tokens{'ClassName'} = $model . 'Model';
  $tokens{'Includes'} = join("\n", map { &Include("$_.cuh") } @classes);
  $tokens{'NodeDeclarations'} = join("\n", map { "$_ " . lcfirst($_) . ';' } @classes);
  foreach $type ('In', 'Ex', 'R', 'F') {
    $tokens{"${type}SpecName"} = "${model}${type}Spec";
    $tokens{"${type}Spec"} = join("\n",
      "BEGIN_NODESPEC($tokens{\"${type}SpecName\"})",
      join("\n", map { "SINGLE_TYPE(1, $_)" } @{$classes{lc($type)}}),
      'END_NODESPEC()');
  }
  
  $source = &ProcessTemplate('ModelHeader', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.cuh") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.cuh");
  print SOURCE $source;
  print SOURCE "\n";
  close SOURCE;
  
  # source file
  my @edges;
  $sth = $dbh->prepare('SELECT ParentNode FROM Edge WHERE ChildNode = ? ' .
      'ORDER BY Position');
  
  foreach $node (@nodes) {
    $sth->execute($node);
    while ($fields = $sth->fetchrow_hashref) {
      push(@edges, "addEdge($$fields{'ParentNode'}, $node);");
    }
    $sth->finish;
  }
    
  $tokens{'NodeDefinitions'} = join("\n  ", map { "addNode($_);" } @nodes);
  $tokens{'EdgeDefinitions'} = join("\n  ", @edges);
  
  $source = &ProcessTemplate('ModelSource', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.cu") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.cu");
  print SOURCE $source;
  print SOURCE "\n";
  close SOURCE;

}

##
## Output source files for nodes.
##
sub OutputNodeSources {
  my $fields;
  my %tokens;
  my $source;

  my $sth = $dbh->prepare('SELECT Name, Formula, Type, Description, ' .
      'LaTeXName, LaTeXFormula ' .
      "FROM Node WHERE Type <> 'Intermediate result' AND Type <> 'Constant'");

  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $tokens{'Name'} = $$fields{'Name'};
    $tokens{'Formula'} = $$fields{'Formula'};
    $tokens{'Description'} = $$fields{'Description'};
    $tokens{'LaTeXName'} = $$fields{'LaTeXName'};
    $tokens{'LaTeXFormula'} = $$fields{'LaTeXFormula'};
    $tokens{'ClassName'} = ucfirst($$fields{'Name'}) . 'Node';
    $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($tokens{'ClassName'}) .
        '_CUH';
    $tokens{'Includes'} = &Includes($$fields{'Name'});
    $tokens{'TraitDeclarations'} = &TraitDeclarations($$fields{'Name'});
    $tokens{'FunctionDeclarations'} = &FunctionDeclarations($$fields{'Name'});
    $tokens{'FunctionDefinitions'} = &FunctionDefinitions($fields);

    $source = &ProcessTemplate('NodeHeader', \%tokens);
    $source = &PrettyPrint($source);

    open(SOURCE, ">$outdir/$tokens{'ClassName'}.cuh") ||
        confess("Could not open $outdir/$tokens{'ClassName'}.cuh");
    print SOURCE $source;
    print SOURCE "\n";
    close SOURCE;
  }
}

##
## Read in all templates.
##
sub GulpTemplates {
  my @files = <$templatedir/*.template>;
  my $file;
  my $name;

  foreach $file (@files) {
    $file =~ /^$templatedir\/(\w+).template$/;
    $name = $1;
    $templates{$name} = &GulpTemplate($file);
  }
}

##
## Read in template from file.
##
## @param file File name.
##
## @return Contents of template file.
##
sub GulpTemplate {
  my $file = shift;
  my $contents;

  {
    undef local $/;
    open(FILE, $file) || confess("Could not read $file");
    $contents = <FILE>;
    close FILE;
  }
  chomp $contents;

  return $contents;
}

##
## Process template.
##
## @param template Template contents.
## @param tokens Hash reference of tokens to replace.
##
## @return Template contents with tokens replaced.
##
sub ProcessTemplate {
  my $template = shift;
  my $tokens = shift;

  my $source = $templates{$template};
  $source =~ s/\<\%\s*(\w+)\s*\%\>/$$tokens{$1}/g;

  return $source;
}

##
## Pretty-print source code.
##
## @param source Source code.
##
## @return Pretty-printed source code.
##
sub PrettyPrint {
  my $source = shift;
  
  $source =~ s/\n{3,}/\n\n/g;
  
  return $source;
}

##
## Check type of node.
##
## @param name Name of node.
##
## @return Type of node -- "in", "ex", "r" or "f".
##
sub NodeType {
  my $name = shift;
  my $result = 'In';
  my $trait;

  my $sth = $dbh->prepare('SELECT Trait FROM NodeTrait WHERE ' .
      "Node = ? AND Trait LIKE 'IS\_%\_NODE'");

  $sth->execute($name);
  $trait = $sth->fetchrow_array;
  if ($trait eq 'IS_IN_NODE') {
    $result = 'in';
  } elsif ($trait eq 'IS_EX_NODE') {
    $result = 'ex';
  } elsif ($trait eq 'IS_R_NODE') {
    $result = 'r';
  } elsif ($trait eq 'IS_F_NODE') {
    $result = 'f';
  } else {
    die("Node $name not of recognised type");
  }
  $sth->finish;

  return $result;
}

##
## Check type of parent relationship.
##
## @param parent Parent name.
## @param child Child name.
##
## @return "inpax", "expax" or "inpax" depending on the type of the parent
## relationship.
##
sub Pax {
  my $parent = shift;
  my $child = shift;

  my $parenttype = &NodeType($parent);
  my $childtype = &NodeType($child);

  if ($parenttype eq 'in') {
    if ($childtype eq 'in') {
      return 'inpax';
    } elsif ($childtype eq 'ex') {
      return 'inpax';
    }
  } elsif ($parenttype eq 'ex') {
    if ($childtype eq 'ex') {
      return 'expax';
    }
  } elsif ($parenttype eq 'r') {
    if ($childtype eq 'in') {
      return 'rpax';
    } elsif ($childtype eq 'ex') {
      return 'rpax';
    }
  } elsif ($parenttype eq 'f') {
    if ($childtype eq 'in') {
      return 'fpax';
    } elsif ($childtype eq 'ex') {
      return 'fpax';
    }
  }
  confess("$parent -> $child, invalid relationship, $parenttype-node -> " .
      $childtype->node);
}

##
## Trait declarations for a node.
##
## @param name Name of node.
##
## @return Trait declarations for inclusion in template.
##
sub TraitDeclarations {
  my $name = shift;
  my @traits;
  my $trait;
  my %tokens;

  my $sth = $dbh->prepare('SELECT Trait FROM NodeTrait WHERE Node = ?');

  $sth->execute($name);
  while ($trait = $sth->fetchrow_array) {
    $tokens{'ClassName'} = ucfirst($name) . 'Node';
    $tokens{'Trait'} = $trait;
    push(@traits, &ProcessTemplate('TraitDeclaration', \%tokens));
  }
  $sth->finish;

  return join("\n", @traits);
}

##
## Function declarations for a node.
##
## @param name Name of node.
##
## @return Function declarations, based on traits, for inclusion in template.
##
sub FunctionDeclarations {
  my $name = shift;
  my @decs;
  my $fields;

  my $sth = $dbh->prepare('SELECT Node, Trait FROM NodeTrait ' .
      'WHERE Node = ?');

  $sth->execute($name);
  while ($fields = $sth->fetchrow_hashref) {
    if (exists $templates{"DEC_$$fields{'Trait'}"}) {
      push(@decs, &ProcessTemplate("DEC_$$fields{'Trait'}", $fields));
    }
  }
  $sth->finish;

  return join("\n", @decs);
}

##
## Function definitions for a node.
##
## @param fields Specification of node.
##
## @return Function definitions, based on traits, for inclusion in template.
##
sub FunctionDefinitions {
  my $fields = shift;
  my @defs;
  my %tokens;
  my $trait;

  my $sth = $dbh->prepare('SELECT Node, Trait FROM NodeTrait ' .
      'WHERE Node = ?');

  $sth->execute($$fields{'Name'});
  while ($trait = $sth->fetchrow_array) { 
    $tokens{'ClassName'} = ucfirst($$fields{'Name'}) . 'Node';
    $tokens{'ParentAliases'} = &ParentAliases($$fields{'Name'});
    $tokens{'ConstantAliases'} = &ConstantAliases($$fields{'Name'});
    $tokens{'Inlines'} = &Inlines($$fields{'Name'});
    $tokens{'Formula'} = $$fields{'Formula'};

    if (exists $templates{"DEF_$trait"}) {
      push(@defs, &ProcessTemplate("DEF_$trait", \%tokens));
    }
  }
  $sth->finish;

  return join("\n\n", @defs);
}

##
## Return parents for node, with inlining.
##
## @param name Name of node.
##
## @returns List of parents, with inlining considered.
##
sub Parents {
  my $name = shift;

  my @results;
  my $parent;
  my $sth = $dbh->prepare('SELECT Name, Formula, Type ' .
    'FROM Node, Edge WHERE Edge.ChildNode = ? AND ' .
    'Edge.ParentNode = Node.Name AND Node.Type <> \'Constant\' ORDER BY ' .
    'Edge.Position');

  $sth->execute($name);
  while ($parent = $sth->fetchrow_hashref) {
    if ($$parent{'Type'} eq 'Intermediate result') {
      push(@results, &Parents($$parent{'Name'})); # inline
    } else {
      push(@results, $$parent{'Name'});
    }
  }
  $sth->finish;

  return uniq(@results);
}

##
## Return parent aliases of node.
##
## @param name Name of node.
##
## @return Parent aliases of node for inclusion in template.
##
sub ParentAliases {
  my $name = shift;
  
  my @parents = &Parents($name);
  my @results;
  my $parent;
  my %tokens;
  my %ctokens;
  my %positions;
  my $fields;
  $positions{'inpax'} = 0;
  $positions{'expax'} = 0;
  $positions{'inpax'} = 0;
  $positions{'rpax'} = 0;
  $positions{'rpax'} = 0;
  $positions{'fpax'} = 0;
  $positions{'fpax'} = 0;
  
  my $sth = $dbh->prepare('SELECT Name, Formula, Type FROM Node WHERE ' .
      'Name = ?');
  
  foreach $parent (@parents) {
    $sth->execute($parent);
    $fields = $sth->fetchrow_hashref;
    if ($$fields{'Type'} eq 'Constant') {
      undef %ctokens;
      $ctokens{'Name'} = $parent;
      $ctokens{'Formula'} = $$fields{'Formula'};
      push(@results, &ProcessTemplate('ConstantAlias', \%ctokens));
    } else {
      $tokens{'Name'} = $parent;
      $tokens{'Pax'} = &Pax($parent, $name);
      $tokens{'Position'} = $positions{$tokens{'Pax'}};
      ++$positions{$tokens{'Pax'}};
      push(@results, &ProcessTemplate('ParentAlias', \%tokens));
    }
    $sth->finish;
  }
  
  return join("\n", @results);
}

##
## Return constant aliases of node.
##
## @param name Name of node.
##
## @return Constant aliases of node for inclusion in template.
##
sub ConstantAliases {
  my $name = shift;

  my @results;
  my $fields;
  my $sth = $dbh->prepare('SELECT Name, Formula AS Value FROM Node, Edge ' .
      'WHERE Node.Type = \'Constant\' AND Edge.ChildNode = ? AND ' .
      'Edge.ParentNode = Node.Name');

  $sth->execute($name); 
  while ($fields = $sth->fetchrow_hashref) {
    push(@results, &ProcessTemplate('ConstantAlias', $fields));
  }
  $sth->finish;
  
  return join("\n", @results);
}

##
## Inline formulas for node.
##
## @param name Name of node.
##
## @return Inline formulas of node, for inclusion in template.
##
sub Inlines {
  my $name = shift;

  my @results;
  my $inline;
  my %tokens;

  my $sth = $dbh->prepare('SELECT Name, Formula ' .
      'FROM Node, Edge WHERE Edge.ChildNode = ? AND ' .
      'Node.Type = \'Intermediate result\' AND ' .
      'Edge.ParentNode = Node.Name ORDER BY Edge.Position');

  $sth->execute($name);
  while ($inline = $sth->fetchrow_hashref) {
    $tokens{'Name'} = $$inline{'Name'};
    $tokens{'Formula'} = $$inline{'Formula'};
    push(@results, &ProcessTemplate('Inline', \%tokens));
  }
  $sth->finish;

  return join("\n", @results);
}

##
## Header includes for node.
##
## @param name Name of node.
##
## @return Header includes for node, for inclusion in template.
##
sub Includes {
  my $name = shift;
  
  my $type = &NodeType($name);
  my @results;
  
  if ($type eq 'in') {
    push(@results, &Include('bi/model/NodeStaticTraits.hpp'));
  } elsif ($type eq 'ex') {
    push(@results, &Include('bi/model/NodeForwardTraits.hpp'));
  } elsif ($type eq 'r') {
    push(@results, &Include('bi/model/NodeRandomTraits.hpp'));
  } elsif ($type eq 'f') {
    #
  } else {
    die("Node $name has unrecognised type");
  }
  push(@results, &Include('bi/model/NodeTypeTraits.hpp'));
  
  return join("\n", @results);
}

##
## Header include.
##
## @param file Header file name.
##
## @return Header include.
##
sub Include {
  my $header = shift;
  
  my %tokens;
  $tokens{'Header'} = $header;
  
  return &ProcessTemplate('Include', \%tokens);
}

__END__

=head1 NAME

sql2cpp -- C++ code generator from SQLite model specification.

=head1 SYNOPSIS

spec2cpp [options]

The SQLite database is read from $model.db, and C++ source files written to 
the $outdir directory, where both $model and $outdir are specified as command
line arguments.

=head1 

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--dbfile>

Specify the database file name.

=item B<--outdir>

Specify the output directory for C++ source files.

=item B<--templatedir>

Specify directory containing C++ templates.

=back

=head1 DESCRIPTION

Reads an SQLite database model specification and generates C++ code
implementing the model for the bi library.

=cut

