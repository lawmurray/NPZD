% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} tabulate_physics ()
%
% Create LaTeX table of hyperparameter estimates for physics GPs.
% @end deftypefn
%
function tabulate_physics ()
  load physics_models.mat

  logs = {
      1;
      1;
      1;
      1;
  };
  display_names = {
    '\\ln MLD';
    '\\ln BCN';
    '\\ln T';
    '\\ln E';
  };
  display_param_names = {
    'a';
    '\varphi';
    'b';
    'c';
    'v';
    'd';
    'w';
    '\sigma^2';
  };
  printf('\\begin{tabular}{rr@{$\\times$}lr@{$\\times$}lr@{$\\times$}lr@{$\\times$}l}\n');
  for i = 1:length(display_names)
      printf(' & \\multicolumn{2}{r}{$%s$}', display_names{i});
  end
  printf(' \\\\\n\\hline\n');
  
  lines = cell(length(display_param_names));
  for i = 1:length(models)
      lines{1} = strcat(lines{1}, sprintf(' & $%.2e$', models{i}.hyp.mean(1)));
      lines{2} = strcat(lines{2}, sprintf(' & $%.2e$', models{i}.hyp.mean(2)));
      lines{3} = strcat(lines{3}, sprintf(' & $%.2e$', models{i}.hyp.mean(3)));
      lines{4} = strcat(lines{4}, sprintf(' & $%.2e$', ...
                                          exp(models{i}.hyp.cov(2))^2));
      lines{5} = strcat(lines{5}, sprintf(' & $%.2e$', ...
                                          exp(models{i}.hyp.cov(1))^2));
      lines{6} = strcat(lines{6}, sprintf(' & $%.2e$', ...
                                          exp(models{i}.hyp.cov(4))^2));
      lines{7} = strcat(lines{7}, sprintf(' & $%.2e$', ...
                                          exp(models{i}.hyp.cov(3))^2));
      lines{8} = strcat(lines{8}, sprintf(' & $%.2e$', ...
                                          exp(models{i}.hyp.lik(1))^2));      
  end
  
  for i = 1:length(display_param_names)
      printf(' $%s$', display_param_names{i});
      line = regexprep(lines{i}, 'e(\-)?\+?0?(\d+)', '$ & $ 10^{$1$2}');
      printf(' %s\\\\\n', line);
  end
  printf('\\end{tabular}\n');
end
