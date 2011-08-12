% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_state ()
%
% Produce plot of state posteriors for NPZD model.
% @end deftypefn
%
function plot_state ()
    MCMC_FILE = 'results/mcmc_acupf.nc.0';
    SIMULATE_FILE = 'results/simulate.nc.0'; % for prior
    OBS_FILE = 'data/C10_TE_obs.nc';
    ps = [25000:50000];
    ns = 2;
    
    nco = netcdf(OBS_FILE, 'r');
    
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
        ts = [1:101];
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
            ts = [2:101]; % first is meaningless
        end

        plot_simulate(SIMULATE_FILE, vars{j}, [], [], ts);
        hold on;
        plot_mcmc(MCMC_FILE, vars{j}, [], ps, ts);
        x = read_var (nco, vars{j}, [], 1, ts);
        if (strcmp(vars{j}, 'N') || strcmp(vars{j}, 'Chla'))
            obsvar = sprintf('%s_obs', vars{j});
            [t y] = read_obs (nco, obsvar, [], ts, ns);
            plot_obs(OBS_FILE, obsvar, [], ts, ns);
        else
            % just need obs times
            [t y] = read_obs (nco, 'N_obs', [], ts, ns);
        end
        plot(ts, x, '.k', 'markersize', 5);
        if (strcmp(vars{j}, 'N') || strcmp(vars{j}, 'Chla'))
            for i = 1:length(t)
                line([t(i), t(i)], [x(i), y(i)], 'linewidth', 1, 'color', 'k');
            end
        end
        hold off;
        plot_defaults;
        %axis([0 100 0 8]);
        ylabel(vars{j});
        
        if strcmp(vars{j}, 'N')
            axis([0 100 50 350]);
            xlabel('Day');
        elseif strcmp(vars{j}, 'P')
            axis([0 100 0 20]);
        elseif strcmp(vars{j}, 'Z')
            axis([0 100 0 20]);
        elseif strcmp(vars{j}, 'D')
            axis([0 100 0 20]);
            xlabel('Day');
        elseif strcmp(vars{j}, 'Chla')
            axis([0 100 0 1]);
            legend(titles);
        end

    end
end
