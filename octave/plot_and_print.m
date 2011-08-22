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
    
    sizes = [ 8 4.5; 8 4.5; 10.5 7; 8 4.5; 10.5 7; 8 4.5 ; 7 10.5; 7 10.5; ...
              8 3.5; 8 3.5];

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
    figure(1);
    subplot(1,2,1);
    plot_converge(0);
    subplot(1,2,2);
    plot_converge(1);
    figure(2);
    plot_state();
    figure(3);
    plot_rstate();
    figure(4);
    plot_state(1);
    figure(5);
    plot_rstate(1);
    figure(6)
    plot_cor(6);
    figure(7);
    plot_parameters(0);
    figure(8);
    plot_parameters(1);
    figure(9);
    hinton_cov(1);
    figure(10);
    hinton_cov(2);
    
    % print
    files = {
        'npzd_converge';
        'npzd_state';
        'npzd_rstate';
        'osp_state';
        'osp_rstate';
        'npzd_cor';
        'npzd_parameters';
        'osp_parameters';
        'npzd_initialcov';
        'npzd_paramcov';
        };

    for i = 1:length (files)
        file = sprintf('%s/%s.pdf', FIG_DIR, files{i});
        saveas(figure(i), file);
        system(sprintf('pdfcrop %s %s', file, file));
    end
end
