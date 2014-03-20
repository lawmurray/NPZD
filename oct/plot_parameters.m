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
    ps = [25000:100000];
    vars = invars();
    names = papernames();
    [mus, sigmas] = priors();
   
    for i = 1:15
        subplot (5,3,i);
        name = nice_name (names{i});
        
        % posterior histogram
        bi_hist('results/posterior.nc', vars{i}, [], ps);
        hold on;
        ax = axis();
        x = linspace(ax(1), ax(2), 500);
        if i == 3
            priorpdf = 'normpdf';
        else
            priorpdf = 'lognpdf';
        end
        bi_plot_prior(x, priorpdf, {mus(i), sigmas(i)}); 
        title(nice_name(names{i}));
    end
end
