% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_rstate (@var{osp})
%
% Produce plot of state posteriors for NPZD model.
%
% @itemize
% @bullet{ @var{osp} True for OSP data, false for twin data.}
% @end itemize
% @end deftypefn
%
function plot_rstate (osp)
    if nargin < 1
        osp = 0;
    end
    
    if osp
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.0');
        SIMULATE_FILE = 'results/simulate.nc.osp'; % for prior
        OBS_FILE = 'data/C10_OSP_71_76_obs.nc';
        ps = [25001:50000];
        ns = 1;
    else
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.*');
        SIMULATE_FILE = 'results/simulate.nc.te'; % for prior
        OBS_FILE = 'data/C10_TE_obs.nc';
        TRUTH_FILE = 'data/C10_TE_true.nc';
        ps = [25001:50000];
        ns = 2;
   end
    
    nco = netcdf(OBS_FILE, 'r');
    if !osp
        nct = netcdf(TRUTH_FILE, 'r');
    end

    titles = {
        '';
    %'Prior';
    %    '';
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
    
    clf;
    for j = 1:length (vars)
        if osp
            ts = [];
        else
            ts = [1:101];
        end
        
        X = [];
        for i = 1:length (MCMC_FILES)
            in = MCMC_FILES{i};
            nci = netcdf(in, 'r');
            t = nci{'time'}(:);
            x = read_var (nci, vars{j}, [], ps, ts);
            X = [ X x ];
            ncclose (nci);
        end
        Q = quantile (X, 0.5, 2);

        subplot(3,3,j);
        %plot_simulate(SIMULATE_FILE, vars{j}, [], [], ts);
        %hold on;
        plot_mcmc(MCMC_FILES, vars{j}, [], ps, ts);
        hold on;
        if !osp
            x = read_var (nct, vars{j}, [], 1, ts);
            [ax, h1, h2] = plotyy(ts, x, ts, Q);
        else
            [ax, h1, h2] = plotyy(t, -10*ones(length (t), 1), t, Q);
        end
        hold off;
        plot_defaults;
        if j >= 7
            xlabel('Day');
        end
        
        set(h1, 'linestyle', 'none', 'marker', '.', 'markersize', 5, 'color', 'k');
        set(h2, 'linewidth', 2, 'color', gray()(32,:));
        set(ax(1), 'ycolor', 'k');
        set(ax(2), 'ycolor', gray()(32,:));
        set(ax(1), 'ylim', [-3 3]);
        set(ax(2), 'ylim', [-0.75 0.75]);
        ylabel(ax(1), vars{j});
        if j == 3
            %legend(ax, titles); % causing errors...
        end
        %axis([0 100 -0.25 0.25]);
    end
end
