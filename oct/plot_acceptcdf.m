% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_acceptcdf (@var{pmatch})
%
% Produce acceptance rate empirical cdfs.
%
% @itemize
% @bullet{ @var{pmatch} True to plot compute matched results, false
% otherwise.}
% @end itemize

% @end deftypefn
%
function plot_acceptcdf (pmatch)
    % arguments
    if nargin < 1
        pmatch = 0;
    end
    
    % plot titles
    titles = {
        'PF0';
        'MUPF0';
        'CUPF0';
        'PF1';
        'MUPF1';
        'CUPF1';
        };
    
    % load models
    load model_acceptance.mat
    
    % subset of models
    if pmatch
        first = 7;
        last = 12;
    else
        first = 1;
        last = 6;
    end

    markers = '.+*ox^';
    hold off;
    x = linspace(0, 1, 1000);
    colour = [
        0 0 0;
        gray()(32,:);
        watercolour(2);
        0 0 0;
        gray()(32,:);
        watercolour(2) ];
    
    % start with empty plots to get correct legend
    for i = first:last
        j = mod(i - 1, 6) + 1;
        fmt = sprintf('--%c', markers(j));
        y = empirical_cdf(x, exp(models{i}.y));
        plot(x(1), y(1), fmt, 'color', colour(j,:), 'linewidth', 3, ...
             'markersize', 4, 'markerfacecolor', colour(j,:));
        hold on;
    end
    
    % plot lines only
    for i = first:last
        j = mod(i - 1, 6) + 1;
        y = empirical_cdf(x, exp(models{i}.y));
        plot(x, y, '--', 'color', colour(j,:), 'linewidth', 3);
    end

    % plot subset of points lines only
    for i = first:last
        j = mod(i - 1, 6) + 1;
        fmt = sprintf('%c', markers(j));
        y = empirical_cdf(x, exp(models{i}.y));
        start = mod(i - 1, 6)*16 + 1;
        plot(x(start:96:end), y(start:96:end), 'color', colour(j,:), fmt, ...
             'linewidth', 3, 'markersize', 4, 'markerfacecolor', ...
             colour(j,:));
    end
   
    plot_defaults;
    ylabel('Cumulative density');
    xlabel('Conditional acceptance rate (CAR)');
    legend(titles, 'location', 'southeast');
    axis([0 1 0 1]);
    set(gca, 'interpreter', 'tex');
end