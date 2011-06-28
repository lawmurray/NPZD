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
    FIG_DIR = strcat(pwd, '/figs');
    
    % output setup
    for i = 1:1
        figure(i, 'visible', 'off');
        h = figure(i);
        set (h, 'papertype', '<custom>');
        set (h, 'paperunits', 'inches');
        set (h, 'papersize', [8.5 3]);
        set (h, 'paperposition', [0,0,[8.5 3]]);
        set (h, 'defaultaxesfontname', 'Helvetica')
        set (h, 'defaultaxesfontsize', 8)
        set (h, 'defaulttextfontname', 'Helvetica')
        set (h, 'defaulttextfontsize', 8)
        orient('landscape');
    end
    
    % plot
    figure(1);
    plot_minmax();
    
    % print
    file = sprintf('%s/npzd_minmax.pdf', FIG_DIR);
    saveas(figure(1), file);
    system(sprintf('pdfcrop %s %s', file, file));
end
