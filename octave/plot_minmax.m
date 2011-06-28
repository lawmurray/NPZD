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
function plot_minmax ()
    experiments = {'bootstrap'};
    M = 100;

    % prepare labels
    vars = invars();
    vars = map(@nice_greek, vars);
    for i = 16:length(vars)
        vars{i} = strcat('{', vars{i}, '_0}');
    end
    
    % plot
    for i = 1:length(experiments)
        experiment = experiments{i};
        file = sprintf('results/likelihood_%s.nc.0', experiment);
        
        model{i} = model_likelihood(file, invars, [], M);
        model{i} = krig_likelihood(model{i}, 1000);
        mn{i} = min_likelihood(model{i}, 100, 1000);
        mx{i} = max_likelihood(model{i}, 100, 1000);
        
        subplot(1, length(experiments), i);
        bar(mn{i} - mx{i}, 'facecolor', watercolour(1), 'edgecolor', ...
            watercolour(1));
        plot_defaults;
        set(gca, 'xtick', 1:length(vars));
        set(gca, 'xticklabel', vars);
        ylabel('z-score difference');
    end
    
    save minmax.mat
end
