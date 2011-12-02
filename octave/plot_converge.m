% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_converge (@var{pmatch})
%
% Produce plot of convergence statistic for NPZD model.
%
% @itemize
% @bullet{ @var{pmatch} True to plot compute matched results, false
% otherwise.}
% @end itemize

% @end deftypefn
%
function plot_converge (pmatch)
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
    load Rp.mat
    
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
    x = [26:50:75000]';
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
        y = Rp{i};
        
        plot(x(1), y(1), fmt, 'color', colour(j,:), 'linewidth', 3, ...
             'markersize', 4, 'markerfacecolor', colour(j,:));
        hold on;
    end
    
    % plot lines only
    for i = first:last
        j = mod(i - 1, 6) + 1;
        y = Rp{i};
        plot(x, y, '--', 'color', colour(j,:), 'linewidth', 3);
    end

    % plot subset of points lines only
    for i = first:last
        j = mod(i - 1, 6) + 1;
        fmt = sprintf('%c', markers(j));
        y = Rp{i};
        plot(x(1:25:end), y(1:25:end), 'color', colour(j,:), fmt, ...
             'linewidth', 3, 'markersize', 4, 'markerfacecolor', ...
             colour(j,:));
    end
   
    plot_defaults;
    %ylabel('{R^p}');
    xlabel('Step');
    legend(titles);
    axis([0 10000 1 2.5]);
    set(gca, 'interpreter', 'tex');
end
