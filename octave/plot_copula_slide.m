% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_copula (@var{pmatch})
%
% Produce plot of variable and acceptance rate copulas.
% @end deftypefn
%
function plot_copula_slide (pmatch)
    INIT_FILE = 'data/C10_TE_init.nc';
    OBS_FILE = 'data/C10_TE_obs.nc';

    if nargin < 1
        pmatch = 0;
    end

    load model_acceptance

    % truth
    vars = invars();
    names = papernames();
    nc = netcdf(INIT_FILE, 'r');
    truth = zeros(1, 30);
    for i = 1:15
        truth(i) = nc{vars{i}}(:);
        if i != 3
            truth(i) = log(truth(i));
        end
    end
    
    % some reordering for presentation
    %is = [25:30 16:24];
    is = [ 1:15 ];
    
    % prepare labels
    vars = invars();
    names = map(@nice_greek, names);
    for i = 1:15
        names1{i} = names{i};
    end
    for i = 16:21
        names2{i - 15} = names{i + 9};
    end
    for i = 22:length(vars)
        names2{i - 15} = names{i - 6};
    end

    % plot
    if pmatch
        experiments = [7:12];
    else
        experiments = [1:6];
    end
    experiments = [6];
    
    % first drawing
    h = ones(length(is), length(experiments));
    maxz = 0.0;
    for k = 1
        experiment = experiments(k);
        y = models{experiment}.y;
        y = empirical_cdf(y, y);
        for i = 1:length(is)
            x = normcdf(models{experiment}.X(:,is(i)));
            h(i,k) = 0.075; %kernel_optimal_bandwidth([x(:) y(:)])
            res = linspace(0, 1, 40);
            [XX YY] = meshgrid(res, res);
            z = kernel_density([XX(:) YY(:)], [x y], h(i,k));
            maxz = max([ maxz; z ]);
        end
    end 
            
    % second drawing
    lvl = linspace(0, maxz, 10);
    
    for k = 1
        experiment = experiments(k);
        t = standardise(truth, models{experiment}.mu, ...
                        models{experiment}.Sigma);
        y = models{experiment}.y;
        y = empirical_cdf(y, y);
        for i = 1:length(is)
            x = normcdf(models{experiment}.X(:,is(i)));
            t(is(i)) = normcdf(t(is(i)));
            res = linspace(0, 1, 100);
            [XX YY] = meshgrid(res, res);
            z = kernel_density([XX(:) YY(:)], [x y], h(i,k));
            ZZ = reshape(z, size(XX));
            
            %subplot(length(is), length(experiments), ...
            %        (i - 1)*length(experiments) + k);
            subplot(3,5,i);
            contourf(XX, YY, ZZ, lvl);
            hold on;
            plot([ t(is(i)) t(is(i)) ]', [0 1]', '-w', 'marker', '.', 'markersize', 10);
            plot_defaults;
            title(names{is(i)});
            caxis([0 maxz]);
            hold off;
        end
    end 
end
