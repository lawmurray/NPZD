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
    INIT_FILE = 'data/C10_TE_init.nc';
    OBS_FILE = 'data/C10_TE_obs.nc';

    if nargin < 1
        experiment = 1;
    end

    titles = {
        'Minimum';
        '';
        'Maximum';
        };
    
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
    
    % min and max    
    mn = mns{experiment};
    mx = mxs{experiment};

    mu = models{experiment}.mu;
    Sigma = models{experiment}.Sigma;
    U = chol(Sigma);
    sd = sqrt(diag(Sigma))';
    
    mn = mn*U;
    mx = mx*U;
    
    mn = mn./repmat(sd, rows(mn), 1);
    mx = mx./repmat(sd, rows(mx), 1);
    truth = (truth - mu)./sd;
    
    % some reordering for presentation
    mn = mn(:,[7:15 1:6 16:30]);
    mx = mx(:,[7:15 1:6 16:30]);
    truth = truth(:,[7:15 1:6 16:30]);
    
    % prepare labels
    vars = invars();
    vars = map(@nice_greek, vars);
    for i = 1:9
        vars1{i} = vars{i + 6};
    end
    for i = 10:15
        vars1{i} = vars{i - 9};
    end
    for i = 16:length(vars)
        vars2{i - 15} = strcat('{', vars{i}, '_0}');
    end

    % plot
    subplot(2,1,1);    
    h = bar([ mn(:,1:15); mx(:,1:15) ]');
    for i = 1:rows (mn)
        set(h(i), 'facecolor', fade(gray()(48,:), 0.5), 'edgecolor', ...
            gray()(48,:));
    end
    for i = 1:rows (mx)
        
        set(h(rows (mn) + i), 'facecolor', fade(watercolour(2), 0.5), ...
            'edgecolor', watercolour(2));
    end
    hold on;
    plot(truth(1:15), '.k', 'markersize', 6);
    hold off;
    plot_defaults;
    axis([axis()(1) axis()(2) -6 6]);
    set(gca, 'xtick', 1:length(vars1));
    set(gca, 'xticklabel', vars1);
    ylabel('Prior z-score');

    subplot(2,1,2);
    h = bar([ mn(:,16:end); mx(:,16:end) ]');
    for i = 1:rows (mn)
        set(h(i), 'facecolor', fade(gray()(48,:), 0.5), 'edgecolor', ...
            gray()(48,:));
    end
    for i = 1:rows (mx)
        set(h(rows (mn) + i), 'facecolor', fade(watercolour(2), 0.5), ...
            'edgecolor', watercolour(2));
    end
    hold on;
    plot(truth(16:end), '.k', 'markersize', 6);
    hold off;
    plot_defaults;
    axis([axis()(1) axis()(2) -6 6]);
    set(gca, 'xtick', 1:length(vars2));
    set(gca, 'xticklabel', vars2);
    legend(titles, 'location', 'southeast');
    ylabel('Prior z-score');
end
