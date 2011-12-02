% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_parameters (@var{osp})
%
% Produce plot of parameter posteriors for NPZD model.
%
% @itemize
% @bullet{ @var{osp} True for OSP data, false for twin data.}
% @end itemize
% @end deftypefn
%
function plot_parameters (osp)
    if nargin < 1
        osp = 0;
    end
    
    if osp
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.*');
        INIT_FILE = '';
        ps = [49001:5:50000];
    else
        MCMC_FILES = glob('results/mcmc_acupf-0.nc.0');
        INIT_FILE = 'data/C10_TE_init.nc';
        ps = [25000:75000];
   end
   titles = {
        'Posterior';
        'Prior';
        'Truth';
   };
   vars = invars();
   names = papernames();
   [mu0, sigma0] = priors();
   
   % truth
   if !osp
       nc = netcdf(INIT_FILE, 'r');
       truth = zeros(1, 30);
       for i = 1:15
           truth(i) = nc{vars{i}}(:);
           if i != 3
               truth(i) = log(truth(i));
           end
       end
   end
   
   % plot
   clf;
   for i = 1:15
       subplot (5,3,i);
       if i == 3
           name = nice_name (names{i});
           logn = 0;
       else
           name = strcat('log(', nice_name (names{i}), ')');
           logn = 1;
       end
       hist_mcmc (MCMC_FILES, vars{i}, [], ps, logn);
       hold on;
       plot_defaults;
 
       % set axes
       xlim = get(gca, 'xlim');
       range = xlim(2) - xlim(1);
       centre = range/2 + xlim(1);
       if osp
           range = range*2.0;
       else
           range = range*1.5;
       end
       xlim = [ centre - range/2, centre + range/2 ];
       xlim = [ min(truth(i), xlim(1)) max(truth(i), xlim(2)) ];
       set(gca, 'xlim', xlim);
 
       % prior curve
       x = linspace (xlim(1), xlim(2), 100);
       y = normpdf (x, mu0(i), sigma0(i));
       plot(x, y, '-', 'color', gray()(32,:), 'linewidth', 3);
       
       % truth
       ax = axis();
       ax = [ax(1) ax(2) ax(3) 1.2*ax(4)];
       axis(ax);
       if !osp
           plot([ truth(i) truth(i) ]', [0 0.99*ax(4)]', '-k', ...
                'marker', '.', 'markersize', 10);
       end
       
       title (name);
       if i == 3
           legend(titles);
       end
       
       % thin out xticklabels for particular plots
       if i == 1 || i == 2 || i == 4 || i == 12 || i == 13 || i == 14
           labels = get(gca, 'xticklabel');
           ticks = get(gca, 'xtick');
           set(gca, 'xticklabel', labels(1:end));
           set(gca, 'xtick', ticks(1:end));
       end
   end
end
