##
## Convert SQLite model specification to LaTeX document.
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
my $templatedir = 'templates/tex';
my $dbfile = "$model.db";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "templatedir=s" => \$templatedir,
    "dbfile=s" => \$dbfile);

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$dbfile", '', '',
    { AutoCommit => 0 });

my %templates;
&GulpTemplates;
&OutputDocument;

##
## Output document.
##
sub OutputDocument {
  my $type;
  my $node;
  my %tokens;
  
  $tokens{'Name'} = $model;
  $tokens{'TypeSections'} = &TypeSections;
  
  print &ProcessTemplate('Document', \%tokens);
  print "\n";
}

##
## Output all type sections.
##
## @return Type sections for inclusion in template.
##
sub TypeSections {
  my @results;
  my %tokens;
  my $type;
  my $sth = $dbh->prepare('SELECT Name, Description FROM NodeType ' .
      'ORDER BY Position');
      
  $sth->execute;
  while ($type = $sth->fetchrow_hashref) {
    $tokens{'Name'} = $$type{'Name'};
    $tokens{'Description'} = $$type{'Description'};
    $tokens{'Nodes'} = &TypeSectionNodes($$type{'Name'});
    push(@results, &ProcessTemplate('TypeSection', \%tokens));
  }
  $sth->finish;
  
  return join("\n", @results);
}

##
## Output nodes for type section.
##
## @param type Node type.
##
## @return List of nodes for inclusion in type section template.
##
sub TypeSectionNodes {
  my $type = shift;
  
  my @results;
  my $tokens;
  my $sth = $dbh->prepare('SELECT Name, Type, Formula, LaTeXName, ' .
      'LaTeXFormula, Description FROM Node ' .
      'WHERE Type = ? ORDER BY Position');
      
  $sth->execute($type);
  while ($tokens = $sth->fetchrow_hashref) {
    if ($$tokens{'LaTeXFormula'} eq '') {
      $$tokens{'LaTeXFormula'} = '-';
    }
    if ($$tokens{'Description'} eq '') {
      $$tokens{'Description'} = '-';
    }
    push(@results, &ProcessTemplate('TypeSectionNode', $tokens));
  }
  $sth->finish;
  
  return join("\n", @results);
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


__END__

=head1 NAME

sql2tex -- LaTeX document from SQLite model specification.

=head1 SYNOPSIS

spec2cpp [options]

The SQLite database is read from $model.db, and LaTeX source written to
stdout, where $model is specified as a command line argument.

=head1

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--dbfile>

Specify the database file name.

=item B<--templatedir>

Specify directory containing LaTeX templates.

=back

=head1 DESCRIPTION

Reads an SQLite database model specification and generates a LaTeX document
describing the model.

=cut
