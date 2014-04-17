% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_and_print ()
%
% Produce plots and print for manuscript.
% @end deftypefn
%
function plot_and_print ()
    FIG_DIR = 'figs';
    mkdir(FIG_DIR);
    
    sz = [ 12 9 ];
    set (figure(1), 'papersize', sz);
    set (figure(1), 'paperposition', [0 0 sz]);

    clf;
    subplot(1,3,1);
    plot_metric(1);
    title('MSE(log z)^{-1} Mean(t)^{-1}');
    xlabel('Bootstrap');
    ylabel('Bridge');
    subplot(1,3,2);
    plot_metric(2);
    title('ESS(z) Mean(t)^{-1}');
    xlabel('Bootstrap');
    subplot(1,3,3);
    plot_metric(3);
    title('CAR(z) Mean(t)^{-1}');
    xlabel('Bootstrap');
    file = sprintf('%s/metrics.pdf', FIG_DIR);
    saveas(figure(1), file);
    system(sprintf('pdfcrop %s %s', file, file));
   
    clf;
    sz = [ 12 9 ];
    set (figure(1), 'papersize', sz);
    set (figure(1), 'paperposition', [0 0 sz]);
    plot_state();
    file = sprintf('%s/state.pdf', FIG_DIR);
    saveas(figure(1), file);
    system(sprintf('pdfcrop %s %s', file, file));
   
    clf;
    sz = [ 12 12 ];
    set (figure(1), 'papersize', sz);
    set (figure(1), 'paperposition', [0 0 sz]);
    plot_parameters();
    file = sprintf('%s/parameters.pdf', FIG_DIR);
    saveas(figure(1), file);
    system(sprintf('pdfcrop %s %s', file, file));
end
