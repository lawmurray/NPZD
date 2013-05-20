% Copyright (C) 2011-2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_slides ()
%
% Produce plots and print for presentation slides.
% @end deftypefn
%
function plot_slides ()
    simulate_file = 'results/simulate_osp.nc';
    sample_file = 'results/sample_osp.nc';
    predict_file = 'results/predict_osp.nc';
    obs_file = 'data/C10_OSP_71_76_obs_pad.nc';
    path_range = [10000:4000:100000];
    quantile_range = [10000:100000];
    
    % incremental path plots
    hold off;
    bi_plot_paths(simulate_file, 'N', [], path_range);
    axis('tight');
    ax = axis();
    axis([ax(1) ax(2) ax(3) 350]);
    xlabel('Days');
    ylabel('N');
    saveas(figure(1), 'figs/npzd_N_prior_paths.pdf');
    hold on;
    bi_plot_paths(obs_file, 'N_obs');
    saveas(figure(1), 'figs/npzd_N_obs.pdf');
    bi_plot_paths(sample_file, 'N', [], path_range);
    bi_plot_paths(obs_file, 'N_obs');
    saveas(figure(1), 'figs/npzd_N_posterior_paths.pdf');
    bi_plot_paths(predict_file, 'N', [], path_range);
    bi_plot_paths(obs_file, 'N_obs');
    saveas(figure(1), 'figs/npzd_N_predict_paths.pdf');
    
    % quantile plots
    vars = {'N'; 'P'; 'Z'; 'D'; 'Chla'; 'N_y'; 'Chla_y'};
    for i = 1:length(vars)
        hold off;
        bi_plot_quantiles(simulate_file, vars{i}, [], quantile_range);
        axis('tight');
        xlabel('Days');
        if strcmp(vars{i}, 'N') || strcmp(vars{i}, 'N_y')
            ax = axis();
            axis([ax(1) ax(2) ax(3) 350]);
            ylabel(vars{i});
            set(get(figure(1), 'currentaxes'), 'yscale', 'linear');
        else
            ylabel(sprintf('log(%s)', vars{i}));
            set(get(figure(1), 'currentaxes'), 'yscale', 'log');
        end
        hold on;
        bi_plot_quantiles(sample_file, vars{i}, [], quantile_range);
        bi_plot_quantiles(predict_file, vars{i}, [], quantile_range);
        if strcmp(vars{i}, 'N') || strcmp(vars{i}, 'N_y')
            bi_plot_paths(obs_file, 'N_obs');
        elseif strcmp(vars{i}, 'Chla') || strcmp(vars{i}, 'Chla_y')
            bi_plot_paths(obs_file, 'Chla_obs');
        end
        saveas(figure(1), sprintf('figs/npzd_%s_quantiles.pdf', vars{i}));
    end
    
    % histogram plots
    params = {'muZCl'; 'muASN'; 'muPgC'; 'muZmQ'};
    priorparams = {
        {log(0.2), 1.3},
        {log(0.3), 1.0},
        {log(1.2), 0.63},
        {log(0.01), 1.0}
    };
    for i = 1:length(params)
        hold off;
        bi_hist(sample_file, params{i}, [], quantile_range);
        hold on;
        x = linspace(axis()(1), axis()(2), 200);
        bi_plot_prior(x, 'lognpdf', priorparams{i});
        xlabel(params{i});
        saveas(figure(1), sprintf('figs/npzd_%s_hist.pdf', params{i}));
    end
end
