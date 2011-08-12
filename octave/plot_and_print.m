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
    
    sizes = [ 8 4.5; 8 4.5; 8 4.5; 8 6 ];

    % output setup
    for i = 1:1
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
    
    % print
    files = {
        'npzd_converge';
        };

    for i = 1:1
        file = sprintf('%s/%s.pdf', FIG_DIR, files{i});
        saveas(figure(1), file);
        system(sprintf('pdfcrop %s %s', file, file));
    end
end
