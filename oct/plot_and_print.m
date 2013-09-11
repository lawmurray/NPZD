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
    
    sizes = [ 8 4.5; 10.5 7; 7 10.5; 7 10.5];

    % output setup
    for i = 1:rows (sizes)
        sz = sizes(i,:);
        figure(i, 'visible', 'off');
        h = figure(i);
        set (h, 'papertype', '<custom>');
        set (h, 'paperunits', 'inches');
        set (h, 'papersize',  sz);
        set (h, 'paperposition', [0,0,sz]);
        set (h, 'defaultaxesfontname', 'Helvetica')
        set (h, 'defaultaxesfontsize', 8)
        set (h, 'defaulttextfontname', 'Helvetica')
        set (h, 'defaulttextfontsize', 8)
        if sz(1) > sz (2)
            orient('landscape');
        else
            orient('portrait');
        end
    end
    
    % plot
    %figure(1);
    %plot_state();
    %figure(2);
    %plot_noise();
    %figure(3);
    %plot_parameters(4);
    figure(4)
    plot_physics;
    
    % print
    files = {
        'state';
        'noise';
        'parameters';
        'physics';
        };

    i = 4;
    %for i = 1:length (files)
        file = sprintf('%s/%s.pdf', FIG_DIR, files{i});
        saveas(figure(i), file);
        system(sprintf('pdfcrop %s %s', file, file));
        %end
end
