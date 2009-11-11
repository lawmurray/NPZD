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
my $templatedir = 'templates/cpp';
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
&OutputModelPrior;
&OutputNodeSources;
&OutputNodePriors;

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
  my @parents;
  my $parent;

  my $sth = $dbh->prepare('SELECT Name FROM Node WHERE ' .
      "Category <> 'Intermediate result' AND Category <> 'Constant' ".
      'ORDER BY Position ASC');
  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $type = &NodeCategory($$fields{'Name'});
    push(@nodes, $$fields{'Name'});
    push(@classes, ucfirst($$fields{'Name'}) . 'Node');
    push(@{$nodes{$type}}, $$fields{'Name'});
    push(@{$classes{$type}}, ucfirst($$fields{'Name'}) . 'Node');
  }
  $sth->finish;

  # header file
  $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($model) . 'MODEL_HPP';
  $tokens{'ClassName'} = $model . 'Model';
  $tokens{'Includes'} = join("\n", map { &Include("$_.hpp") } @classes);
  $tokens{'NodeDeclarations'} = join("\n  ", map { ucfirst($_) . 'Node ' . $_ . ';' } @nodes);
  foreach $type ('S', 'D', 'C', 'R', 'F', 'O', 'P') {
    $tokens{"${type}TypeListName"} = "${model}${type}TypeList";
    $tokens{"${type}TypeList"} = join("\n",
      "BEGIN_TYPELIST($tokens{\"${type}TypeListName\"})",
      join("\n", map { "SINGLE_TYPE(1, $_)" } @{$classes{lc($type)}}),
      'END_TYPELIST()');
  }

  $source = &ProcessTemplate('ModelHeader', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.hpp") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.hpp");
  print SOURCE $source;
  print SOURCE "\n";
  close SOURCE;

  # source file
  my @edges;

  foreach $node (@nodes) {
    @parents = &Parents($node);
    foreach $parent (@parents) {
      push(@edges, "addArc($parent, $node);");
    }
  }

  $tokens{'NodeDefinitions'} = join("\n  ", map { "addNode(${_});" } @nodes);
  $tokens{'EdgeDefinitions'} = join("\n  ", @edges);

  $source = &ProcessTemplate('ModelSource', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.cpp") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.cpp");
  print SOURCE $source;
  print SOURCE "\n";
  close SOURCE;

}

##
## Output source files for prior.
##
sub OutputModelPrior {
  my $fields;
  my %tokens;
  my @nodes;
  my @classes;
  my %nodes;
  my %classes;
  my %priors;
  my $type;
  my $prior;
  my $node;
  my $source;
  my @parents;
  my $parent;
  my $i;

  my $sth = $dbh->prepare('SELECT Name ' .
      "FROM Node, NodeTrait WHERE Category <> 'Intermediate result' AND " .
      "Category <> 'Constant' AND Trait LIKE '\%PRIOR\%' AND " .
      'NodeTrait.Node = Node.Name ORDER BY Position ASC');

  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $type = &NodeCategory($$fields{'Name'});
    if ($type =~ /^[sdcp]$/) {
      $prior = &NodePrior($$fields{'Name'});
    } else {
      $prior = '';
    }
    push(@nodes, $$fields{'Name'});
    push(@classes, ucfirst($$fields{'Name'}) . 'Prior');
    push(@{$nodes{$type}}, $$fields{'Name'});
    push(@{$classes{$type}}, ucfirst($$fields{'Name'}) . 'Prior');
    push(@{$priors{$type}}, $prior);
  }
  $sth->finish;

  # header file
  $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($model) . 'PRIOR_HPP';
  $tokens{'ClassName'} = $model . 'Prior';
  $tokens{'Includes'} = join("\n", map { &Include("${_}.hpp") } @classes);
  $tokens{'NodeDeclarations'} = join("\n  ", map { ucfirst($_) . 'Prior ' . $_ . ';' } @nodes);

  foreach $type ('S', 'D', 'C', 'P') {
    $tokens{"${type}PriorListName"} = "${model}${type}PriorList";
    $tokens{"${type}PriorList"} = join("\n",
      "BEGIN_TYPELIST($tokens{\"${type}PriorListName\"})",
      &PriorList($priors{lc($type)}),
      'END_TYPELIST()');
  }

  $source = &ProcessTemplate('ModelPriorHeader', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.hpp") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.hpp");
  print SOURCE $source;
  print SOURCE "\n";
  close SOURCE;

  # source file
  foreach $type ('s', 'd', 'c', 'p') {
    $i = 0;
    foreach $node (@{$nodes{$type}}) {
      if (&NodePrior($node) ne '') {
        $tokens{uc($type) . 'PriorDefinitions'} .= "  ${type}0.set($i, $node.getPrior());\n";
      }
      $i++;
    }
  }

  $source = &ProcessTemplate('ModelPriorSource', \%tokens);
  $source = &PrettyPrint($source);

  open(SOURCE, ">$outdir/$tokens{'ClassName'}.cpp") ||
      confess("Could not open $outdir/$tokens{'ClassName'}.cpp");
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

  my $sth = $dbh->prepare('SELECT Name, LaTeXName, Category, Description FROM Node ' .
      "WHERE Category <> 'Intermediate result' AND Category <> 'Constant'");

  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $tokens{'Name'} = $$fields{'Name'};
    $tokens{'Formula'} = $$fields{'Formula'};
    $tokens{'Description'} = $$fields{'Description'};
    $tokens{'LaTeXName'} = $$fields{'LaTeXName'};
    $tokens{'ClassName'} = ucfirst($$fields{'Name'}) . 'Node';
    $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($tokens{'ClassName'}) .
        '_HPP';
    $tokens{'Includes'} = &Includes($$fields{'Name'});
    $tokens{'TraitDeclarations'} = &TraitDeclarations($$fields{'Name'});
    $tokens{'FunctionDeclarations'} = &FunctionDeclarations($$fields{'Name'}, 0);
    $tokens{'FunctionDefinitions'} = &FunctionDefinitions($$fields{'Name'}, 0);

    $source = &ProcessTemplate('NodeHeader', \%tokens);
    $source = &PrettyPrint($source);

    open(SOURCE, ">$outdir/$tokens{'ClassName'}.hpp") ||
        confess("Could not open $outdir/$tokens{'ClassName'}.hpp");
    print SOURCE $source;
    print SOURCE "\n";
    close SOURCE;

    $source = &ProcessTemplate('NodeSource', \%tokens);
    $source = &PrettyPrint($source);

    open(SOURCE, ">$outdir/$tokens{'ClassName'}.cpp") ||
        confess("Could not open $outdir/$tokens{'ClassName'}.cpp");
    print SOURCE $source;
    print SOURCE "\n";
    close SOURCE;
  }
}

##
## Output source files for node priors.
##
sub OutputNodePriors {
  my $fields;
  my %tokens;
  my $source;
  my $key;

  my $sth = $dbh->prepare('SELECT Name, LaTeXName, Category, Description ' .
      "FROM Node, NodeTrait WHERE Category <> 'Intermediate result' AND " .
      "Category <> 'Constant' AND Trait LIKE '\%PRIOR\%' AND " .
      'NodeTrait.Node = Node.Name');

  $sth->execute;
  while ($fields = $sth->fetchrow_hashref) {
    $tokens{'Name'} = $$fields{'Name'};
    $tokens{'Formula'} = $$fields{'Formula'};
    $tokens{'Description'} = $$fields{'Description'};
    $tokens{'LaTeXName'} = $$fields{'LaTeXName'};
    $tokens{'ClassName'} = ucfirst($$fields{'Name'}) . 'Prior';
    $tokens{'Guard'} = 'BIM_' . uc($model) . '_' . uc($tokens{'ClassName'}) .
        '_HPP';
    #$tokens{'Includes'} = &PriorIncludes($$fields{'Name'});
    $tokens{'FunctionDeclarations'} = &FunctionDeclarations($$fields{'Name'}, 1);
    $tokens{'FunctionDefinitions'} = &FunctionDefinitions($$fields{'Name'}, 1);
    $tokens{'PriorType'} = &NodePriorType($$fields{'Name'});

    $source = &ProcessTemplate('NodePriorHeader', \%tokens);
    $source = &PrettyPrint($source);

    open(SOURCE, ">$outdir/$tokens{'ClassName'}.hpp") ||
        confess("Could not open $outdir/$tokens{'ClassName'}.hpp");
    print SOURCE $source;
    print SOURCE "\n";
    close SOURCE;

    $source = &ProcessTemplate('NodePriorSource', \%tokens);
    $source = &PrettyPrint($source);

    open(SOURCE, ">$outdir/$tokens{'ClassName'}.cpp") ||
        confess("Could not open $outdir/$tokens{'ClassName'}.cpp");
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
## @param template Template name.
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
## @return Category of node -- "s", "d", "c", "r", "f", "o" or "p".
##
sub NodeCategory {
  my $name = shift;
  my $result;
  my $trait;

  my $sth = $dbh->prepare('SELECT Trait FROM NodeTrait WHERE ' .
      "Node = ? AND Trait LIKE 'IS\_%\_NODE'");

  $sth->execute($name);
  $trait = $sth->fetchrow_array;
  if ($trait eq 'IS_S_NODE') {
    $result = 's';
  } elsif ($trait eq 'IS_D_NODE') {
    $result = 'd';
  } elsif ($trait eq 'IS_C_NODE') {
    $result = 'c';
  } elsif ($trait eq 'IS_R_NODE') {
    $result = 'r';
  } elsif ($trait eq 'IS_F_NODE') {
    $result = 'f';
  } elsif ($trait eq 'IS_O_NODE') {
    $result = 'o';
  } elsif ($trait eq 'IS_P_NODE') {
    $result = 'p';
  } else {
    die("Node $name not of recognised type");
  }
  $sth->finish;

  return $result;
}

##
## Check prior type of node.
##
## @param name Name of node.
##
## @return Prior type.
##
sub NodePrior {
  my $name = shift;
  my $result;
  my $trait;

  my $sth = $dbh->prepare('SELECT Trait FROM NodeTrait WHERE ' .
      "Node = ? AND Trait LIKE 'HAS\_%\_PRIOR'");

  $sth->execute($name);
  $trait = $sth->fetchrow_array;
  if ($trait eq 'HAS_GAUSSIAN_PRIOR' || $trait eq 'HAS_NORMAL_PRIOR') {
    $result = 'Gaussian';
  } elsif ($trait eq 'HAS_LOG_NORMAL_PRIOR') {
    $result = 'LogNormal';
  } else {
    warn("Node $name does not have prior of recognised type");
    $result = '';
  }
  $sth->finish;

  return $result;
}

##
## Get prior type of node.
##
## @param name Name of node.
##
## @return Prior type.
##
sub NodePriorType {
  my $name = shift;
  my $result;
  my $trait;

  my $sth = $dbh->prepare('SELECT Trait FROM NodeTrait WHERE ' .
      "Node = ? AND Trait LIKE 'HAS\_%\_PRIOR'");

  $sth->execute($name);
  $trait = $sth->fetchrow_array;
  if ($trait eq 'HAS_GAUSSIAN_PRIOR' || $trait eq 'HAS_NORMAL_PRIOR') {
    $result = 'bi::GaussianPdf<>';
  } elsif ($trait eq 'HAS_LOG_NORMAL_PRIOR') {
    $result = 'bi::LogNormalPdf<>';
  } else {
    warn("Node $name does not have prior of recognised type");
    $result = '';
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
## @return "spax", "dpax", "cpax", etc depending on the type of the parent
## relationship.
##
sub Pax {
  my $parent = shift;
  my $child = shift;

  my $parenttype = &NodeCategory($parent);
  my $childtype = &NodeCategory($child);

  if ($parenttype eq 'o') {
    confess("$parent -> $child, invalid relationship, $parenttype-node -> " .
        $childtype->node);
  }

  return "pax.$parenttype";
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
## @param prior True to keep prior-related functions, false to keep others.
##
## @return Function declarations for inclusion in template.
##
sub FunctionDeclarations {
  my $name = shift;
  my $prior = shift;
  my @decs;
  my %tokens;
  my $template;
  my $fields;
  my $continue;

  my $sth = $dbh->prepare('SELECT Function, Formula, LaTeXFormula FROM NodeFormula ' .
      'WHERE Node = ?');

  $sth->execute($name);
  while ($fields = $sth->fetchrow_hashref) {
    $continue = ($$fields{'Function'} eq 'mu0' || $$fields{'Function'} eq 'sigma0');
    if (($continue && $prior) || (!$continue && !$prior)) {
      if ($prior) {
        $tokens{'ClassName'} = ucfirst($name) . 'Prior';
      } else {
        $tokens{'ClassName'} = ucfirst($name) . 'Node';
      }
      $tokens{'ParentAliases'} = &ParentAliases($name);
      $tokens{'ConstantAliases'} = &ConstantAliases($name);
      $tokens{'InlineAliases'} = &InlineAliases($name);
      $tokens{'Formula'} = $$fields{'Formula'};
      $tokens{'LaTeXFormula'} = $$fields{'LaTeXFormula'};

      $template = 'FunctionDeclaration' . ucfirst($$fields{'Function'});
      push(@decs, &ProcessTemplate($template, \%tokens));
    }
  }
  $sth->finish;

  return join("\n\n", @decs);
}

##
## Function definitions for a node.
##
## @param name Name of node.
## @param prior True to keep prior-related functions, false to keep others.
##
## @return Function definitions for inclusion in template.
##
sub FunctionDefinitions {
  my $name = shift;
  my $prior = shift;
  my @defs;
  my %tokens;
  my $template;
  my $fields;
  my $continue;

  my $sth = $dbh->prepare('SELECT Function, Formula, LaTeXFormula FROM NodeFormula ' .
      'WHERE Node = ?');

  $sth->execute($name);
  while ($fields = $sth->fetchrow_hashref) {
    $continue = ($$fields{'Function'} eq 'mu0' || $$fields{'Function'} eq 'sigma0');
    if (($continue && $prior) || (!$continue && !$prior)) {
      if ($prior) {
        $tokens{'ClassName'} = ucfirst($name) . 'Prior';
      } else {
        $tokens{'ClassName'} = ucfirst($name) . 'Node';
      }
      $tokens{'ParentAliases'} = &ParentAliases($name);
      $tokens{'ConstantAliases'} = &ConstantAliases($name);
      $tokens{'InlineAliases'} = &InlineAliases($name);
      $tokens{'Formula'} = $$fields{'Formula'};
      $tokens{'LaTeXFormula'} = $$fields{'LaTeXFormula'};

      $template = 'FunctionDefinition' . ucfirst($$fields{'Function'});
      push(@defs, &ProcessTemplate($template, \%tokens));
    }
  }
  $sth->finish;

  return join("\n\n", @defs);
}

##
## Prior type list for model.
##
## @param Individual types.
##
## @return Type list for inclusion in template.
##
sub PriorList {
  my $types = shift;
  my $type;
  my $lasttype = '';
  my $reps = 0;
  my @result;

  foreach $type (@{$types}) {
    if ($lasttype eq '') {
      $lasttype = $type;
      $reps = 1;
    } elsif ($type eq $lasttype) {
      $reps++;
    } else {
      push(@result, "SINGLE_TYPE($reps, bi::${lasttype}Pdf<>)");
      $lasttype = $type;
      $reps = 1;
    }
  }
  if ($lasttype ne '') {
    push(@result, "SINGLE_TYPE($reps, bi::${lasttype}Pdf<>)");
  }

  return join("\n", @result);
}

##
## Prior declaration for a node.
##
## @param name Name of node.
##
## @return Prior declaration for inclusion in template.
##
sub PriorDeclaration {
  my $name = shift;
  my %tokens;
  my $template;

  $tokens{'ClassName'} = ucfirst($name) . 'Prior';
  $template = 'Prior' . &NodePrior($name) . 'Declaration';

  return &ProcessTemplate($template, \%tokens);
}

##
## Function definitions for a node.
##
## @param name Name of node.
##
## @return Function definitions for inclusion in template.
##
sub PriorDefinition {
  my $name = shift;
  my %tokens;
  my $template;

  $tokens{'ClassName'} = ucfirst($name) . 'Prior';
  $template = 'Prior' . &NodePrior($name) . 'Definition';

  return &ProcessTemplate($template, \%tokens);
}

##
##
##
sub PriorDeclarations {
  my $name = shift;
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
  my $sth = $dbh->prepare('SELECT Name, Category ' .
    'FROM Node, Edge WHERE Edge.ChildNode = ? AND ' .
    'Edge.ParentNode = Node.Name AND Node.Category <> \'Constant\' ORDER BY ' .
    'Edge.Position');

  $sth->execute($name);
  while ($parent = $sth->fetchrow_hashref) {
    if ($$parent{'Category'} eq 'Intermediate result') {
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
  $positions{'pax.s'} = 0;
  $positions{'pax.d'} = 0;
  $positions{'pax.c'} = 0;
  $positions{'pax.r'} = 0;
  $positions{'pax.f'} = 0;
  $positions{'pax.p'} = 0;

  my $sth = $dbh->prepare('SELECT Name, Category FROM Node WHERE ' .
      'Name = ?');
  my $sthc = $dbh->prepare('SELECT Formula FROM NodeFormula WHERE ' .
      "Node = ? AND Function = 'x'");

  foreach $parent (@parents) {
    $sth->execute($parent);
    $fields = $sth->fetchrow_hashref;
    if ($$fields{'Category'} eq 'Constant') {
      undef %ctokens;
      $sthc->execute($parent);
      $ctokens{'Name'} = $parent;
      $ctokens{'Formula'} = $sthc->fetchrow_array;
      $sthc->finish;
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
## Return constant parents for node, with inlining.
##
## @param name Name of node.
##
## @returns List of inlines.
##
sub Constants {
  my $name = shift;

  my @results;
  my $fields;
  my $sth = $dbh->prepare('SELECT Name, Category FROM Node, Edge ' .
      'WHERE (Category = \'Constant\' OR Category = \'Intermediate result\') AND ' .
      'Edge.ChildNode = ? AND Edge.ParentNode = Node.Name');

  $sth->execute($name);
  while ($fields = $sth->fetchrow_hashref) {
    if ($$fields{'Category'} eq 'Constant') {
      push(@results, $$fields{'Name'});
    } else {
      push(@results, &Constants($$fields{'Name'}));
    }
  }
  $sth->finish;

  return uniq(@results);
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

  my @constants = &Constants($name);
  my @results;
  my $constant;
  my $fields;
  my $sth = $dbh->prepare('SELECT Node AS Name, Formula AS Value FROM ' .
      "NodeFormula WHERE Node = ? AND Function = 'x'");

  foreach $constant (@constants) {
    $sth->execute($constant);
    while ($fields = $sth->fetchrow_hashref) {
      push(@results, &ProcessTemplate('ConstantAlias', $fields));
    }
    $sth->finish;
  }

  return join("\n", @results);
}

##
## Return inlines for node, recursively.
##
## @param name Name of node.
##
## @returns List of inlines.
##
sub Inlines {
  my $name = shift;

  my @results;
  my $inline;
  my $sth = $dbh->prepare('SELECT Name ' .
      'FROM Node, Edge WHERE Edge.ChildNode = ? AND ' .
      'Node.Category = \'Intermediate result\' AND ' .
      'Edge.ParentNode = Node.Name ORDER BY Edge.Position');

  $sth->execute($name);
  while ($inline = $sth->fetchrow_array) {
    push(@results, &Inlines($inline)); # recursive inline
    push(@results, $inline);
  }
  $sth->finish;

  return uniq(@results);
}

##
## Inline formulas for node.
##
## @param name Name of node.
##
## @return Inline formulas of node, for inclusion in template.
##
sub InlineAliases {
  my $name = shift;

  my @results;
  my @inlines = &Inlines($name);
  my $fields;
  my $inline;

  my $sth = $dbh->prepare('SELECT Node AS Name, Formula FROM NodeFormula ' .
      "WHERE Node = ? AND Function = 'x'");

  foreach $inline (@inlines) {
    $sth->execute($inline);
    $fields = $sth->fetchrow_hashref;
    push(@results, &ProcessTemplate('InlineAlias', $fields));
    $sth->finish;
  }

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

  my $type = &NodeCategory($name);
  my @results;

  push(@results, &Include('bi/traits/type_traits.hpp'));
  if ($type eq 's') {
    push(@results, &Include('bi/traits/static_traits.hpp'));
    push(@results, &Include('bi/traits/prior_traits.hpp'));
  } elsif ($type eq 'd' || $type eq 'c') {
    push(@results, &Include('bi/traits/forward_traits.hpp'));
    push(@results, &Include('bi/traits/prior_traits.hpp'));
  } elsif ($type eq 'r') {
    push(@results, &Include('bi/traits/random_traits.hpp'));
  } elsif ($type eq 'f') {
    #
  } elsif ($type eq 'o') {
    push(@results, &Include('bi/traits/likelihood_traits.hpp'));
  } elsif ($type eq 'p') {
    push(@results, &Include('bi/traits/prior_traits.hpp'));
  } else {
    die("Node $name has unrecognised type");
  }

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
