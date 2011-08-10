% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_minmax ()
%
% Produce comparison plot of global minimum and maximum in NPZD model.
% @end deftypefn
%
function plot_minmax (experiment)
    if nargin < 1
        experiment = 1;
    end
    
    load model_acceptance

    % prepare labels
    vars = invars();
    vars = map(@nice_greek, vars);
    for i = 16:length(vars)
        vars{i} = strcat('{', vars{i}, '_0}');
    end
    
    mn = mns{experiment}(1,:);
    mx = mxs{experiment}(1,:);
    
    Sigma = models{experiment}.Sigma;
    U = chol(Sigma);
    sd = sqrt(diag(Sigma))';
    
    mn = mn*U;
    mx = mx*U;
    
    mn = mn./sd;
    mx = mx./sd;
    
    % plot
    bar(mn - mx, 'facecolor', fade(watercolour(2), 0.5), 'edgecolor', ...
        watercolour(2));
    plot_defaults;
    set(gca, 'xtick', 1:length(vars));
    set(gca, 'xticklabel', vars);
    ylabel('z-score difference');
end
