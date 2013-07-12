% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_cor ()
%
% Produce plot of variable and acceptance rate correlations.
% @end deftypefn
%
function plot_cor (experiment)
    INIT_FILE = 'data/C10_TE_init.nc';
    OBS_FILE = 'data/C10_TE_obs.nc';

    if nargin < 1
        experiment = 1;
    end

    load model_acceptance

    % truth
    vars = invars();
    nc = netcdf(INIT_FILE, 'r');
    truth = zeros(1, 30);
    for i = 1:30
        truth(i) = nc{vars{i}}(:);
        if i != 3
            truth(i) = log(truth(i));
        end
    end

    % support
    mu = models{experiment}.mu;
    Sigma = models{experiment}.Sigma;
    X = models{experiment}.X;
    X = unstandardise(X, mu, Sigma);
    Y = X;
    for i = 1:columns (X)
        if i != 3
            Y(:,i) = exp(Y(:,i));
        end
    end
    y = models{experiment}.y;
    C1 = cor(X, y);
    C2 = cor(Y, y);
    
    % some reordering for presentation
    is = [1:15 25:30 16:24];
    C1 = C1(is);
    C2 = C2(is);
    truth = truth(is);
    
    % prepare labels
    vars = invars();
    vars = map(@nice_greek, vars);
    for i = 1:15
        vars1{i} = vars{i};
    end
    for i = 16:21
        vars2{i - 15} = strcat ('{', vars{i + 9}, '_0}');
    end
    for i = 22:length(vars)
        vars2{i - 15} = strcat ('{', vars{i - 6}, '_0}');
    end

    % plot
    subplot(2,1,1);    
    h = bar([ C1(1:15) C2(1:15) ]);
    set(h(1), 'facecolor', fade(watercolour(2), 0.5), ...
        'edgecolor', watercolour(2));
    set(h(2), 'facecolor', fade(gray()(36,:), 0.5), ...
        'edgecolor', gray()(36,:));
    plot_defaults;
    axis([ axis()(1:2) -0.4 0.4 ]);
    set(gca, 'xtick', 1:length(vars1));
    set(gca, 'xticklabel', vars1);
    legend({
        'Cor(x,\alpha)';
        'Cor(\log x,\alpha)';
        });
    
    subplot(2,1,2);
    h = bar([ C1(16:end) C2(16:end) ]);
    set(h(1), 'facecolor', fade(watercolour(2), 0.5), ...
        'edgecolor', watercolour(2));
    set(h(2), 'facecolor', fade(gray()(36,:), 0.5), ...
        'edgecolor', gray()(36,:));
    plot_defaults;
    axis([ axis()(1:2) -0.4 0.4 ]);
    set(gca, 'xtick', 1:length(vars2));
    set(gca, 'xticklabel', vars2);
end
