% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_rstate ()
%
% Produce plot of state posteriors for NPZD model.
% @end deftypefn
%
function plot_rstate ()
    MCMC_FILE = 'results/mcmc_acupf.nc.0';
    SIMULATE_FILE = 'results/simulate.nc.0'; % for prior
    OBS_FILE = 'data/C10_TE_obs.nc';
    ps = [25000:50000];
    
    nci = netcdf(MCMC_FILE, 'r');
    nco = netcdf(OBS_FILE, 'r');
    
    titles = {
        '';
        'Prior';
        '';
        'Posterior';
        'Truth';
        'Posterior mean, right y-axis';
        };
    
    vars = {
        'rPgC';
        'rPCh';
        'rPRN';
        'rASN';
        'rZin';
        'rZCl';
        'rZgE';
        'rDre';
        'rZmQ';
        };
    
    for j = 1:length (vars)
        ts = [1:101];
        X = read_var (nci, vars{j}, [], ps, ts);
        Q = quantile (X, 0.5, 2);
        
        subplot(3,3,j);
        plot_simulate(SIMULATE_FILE, vars{j}, [], [], ts);
        hold on;
        plot_mcmc(MCMC_FILE, vars{j}, [], ps, ts);
        x = read_var (nco, vars{j}, [], 1, ts);
        [ax, h1, h2] = plotyy(ts, x, ts, Q);
        hold off;
        plot_defaults;
        set(h1, 'linestyle', 'none', 'marker', '.', 'markersize', 5, 'color', 'k');
        set(h2, 'linewidth', 2, 'color', gray()(32,:));
        set(ax(1), 'ycolor', 'k');
        set(ax(2), 'ycolor', gray()(32,:));
        set(ax(2), 'ylim', [-0.25 0.25]);
        ylabel(ax(1), vars{j});
        if j == 3
            %legend(ax, titles); % causing errors...
        end
        %axis([0 100 -0.25 0.25]);
    end
end
