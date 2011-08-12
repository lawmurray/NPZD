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
    
    sizes = [ 8 4.5; 8 4.5; 10.5 7; 8 4.5 ];

    % output setup
    for i = 1:4
        figure(i, 'visible', 'off');
        h = figure(i);
        set (h, 'papertype', '<custom>');
        set (h, 'paperunits', 'inches');
        set (h, 'papersize',  sizes(i,:));
        set (h, 'paperposition', [0,0,sizes(i,:)]);
        set (h, 'defaultaxesfontname', 'Helvetica')
        set (h, 'defaultaxesfontsize', 8)
        set (h, 'defaulttextfontname', 'Helvetica')
        set (h, 'defaulttextfontsize', 8)
        orient('landscape');
    end
    
    % plot
    figure(1);
    subplot(1,2,1);
    plot_converge(0);
    subplot(1,2,2);
    plot_converge(1);
    figure(2);
    plot_state();
    figure(3);
    plot_rstate();
    figure(4)
    plot_minmax(6);
    
    % print
    files = {
        'npzd_converge';
        'npzd_state';
        'npzd_rstate';
        'npzd_minmax';
        };

    for i = 1:length (files)
        file = sprintf('%s/%s.pdf', FIG_DIR, files{i});
        saveas(figure(i), file);
        system(sprintf('pdfcrop %s %s', file, file));
    end
end
