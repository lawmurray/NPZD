% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_state (@var{osp})
%
% Produce plot of state posteriors for NPZD model.
%
% @itemize
% @bullet{ @var{osp} True for OSP data, false for twin data.}
% @end itemize
% @end deftypefn
%
function plot_state (osp)
    if nargin < 1
        osp = 0;
    end
    
    if osp
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.0');
        SIMULATE_FILE = 'results/simulate.nc.osp'; % for prior
        OBS_FILE = 'data/C10_OSP_71_76_obs_pad.nc';
        ps = [25001:50000];
        ns = 1;
    else
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.*');
        SIMULATE_FILE = 'results/simulate.nc.te'; % for prior
        TRUTH_FILE = 'data/C10_TE_true.nc';
        OBS_FILE = 'data/C10_TE_obs.nc';
        ps = [25001:50000];
        ns = 2;
   end
    
    nco = netcdf(OBS_FILE, 'r');
    if !osp
        nct = netcdf(TRUTH_FILE, 'r');
    end
    
    titles = {
        '';
        'Prior';
        '';
        'Posterior';
        'Observed';
        'Truth';
        };
    
    vars = {
        'N';
        'P';
        'Z';
        'D';
        'Chla';
        };
    
    for j = 1:length (vars)
        if osp
            ts = [];
        else
            ts = [1:101];
        end
        if strcmp(vars{j}, 'N')
            subplot(2,2,4);
        elseif strcmp(vars{j}, 'P')
            subplot(3,2,1);
        elseif strcmp(vars{j}, 'Z')
            subplot(3,2,3);
        elseif strcmp(vars{j}, 'D')
            subplot(3,2,5);
        elseif strcmp(vars{j}, 'Chla')
            subplot(2,2,2);
            if !osp
                ts = [2:101]; % first is meaningless
            end
        end

        plot_simulate(SIMULATE_FILE, vars{j}, [], [], ts);
        hold on;
        plot_mcmc(MCMC_FILES, vars{j}, [], ps, ts);
        if !osp
            x = read_var (nct, vars{j}, [], 1, ts);
        end
        if (strcmp(vars{j}, 'N') || strcmp(vars{j}, 'Chla'))
            obsvar = sprintf('%s_obs', vars{j});
            [t y] = read_obs (nco, obsvar, [], ts, ns);
            plot_obs(OBS_FILE, obsvar, [], ts, ns);
        else
            % just need obs times
            [t y] = read_obs (nco, 'N_obs', [], ts, ns);
        end
        if !osp
            plot(ts, x, '.k', 'markersize', 5);
            if (strcmp(vars{j}, 'N') || strcmp(vars{j}, 'Chla'))
                for i = 1:length(t)
                    line([t(i), t(i)], [x(i), y(i)], 'linewidth', 1, ...
                        'color', 'k');
                end
            end
        end
        hold off;
        plot_defaults;
        ylabel(vars{j});
        ax = axis();
        
        if strcmp(vars{j}, 'N')
            if osp
                ymin = 0;
                ymax = 300;
            else
                ymin = 50;
                ymax = 450;
            end
            axis([0 ax(2) ymin ymax]);
            xlabel('Day');
        elseif strcmp(vars{j}, 'P')
            if osp
                ymax = 60;
            else
                ymax = 20;
            end
            axis([0 ax(2) 0 ymax]);
        elseif strcmp(vars{j}, 'Z')
            if osp
                ymax = 40;
            else
                ymax = 20;
            end
            axis([0 ax(2) 0 ymax]);
        elseif strcmp(vars{j}, 'D')
            if osp
                ymax = 60;
            else
                ymax = 20;
            end
            axis([0 ax(2) 0 ymax]);
            xlabel('Day');
        elseif strcmp(vars{j}, 'Chla')
            if osp
                ymax = 2;
            else
                ymax = 1.2;
            end
            axis([0 ax(2) 0 ymax]);
            legend(titles, 'location', 'northwest');
        end

    end
end
