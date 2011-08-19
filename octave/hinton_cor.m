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
function hinton_cor ()
    MCMC_FILES = glob('results/mcmc_*pf-*.nc.*');
    URTS_FILE = 'results/urts.nc.0';
    ps = [40001:1:50000];
    %vars = invars();
        vars = {
                'PgC';
        'PCh';
        'PRN';
        'ASN';
        'Zin';
        'ZCl';
        'ZgE';
        'Dre';
        'ZmQ';
        'EZ';
        'Chla';
        'P';
        'Z';
        'D';
        'N';

        'Kw';
        'KCh';
        'Dsi';
        'ZgD';
        'PDF';
        'ZDF';
        'muPgC';
        'muPCh';
        'muPRN';
        'muASN';
        'muZin';
        'muZCl';
        'muZgE';
        'muDre';
        'muZmQ';
    };

    is = [1:15 25:39];
    logs = ones (30, 1);
    logs(18) = 0;

    names = vars;
    for i = 1:15
        names{i} = strcat ('{', vars{i}, '_0}');
    end
    names = cellfun(@nice_name, names, 'UniformOutput', 0);
    
    clf;
    colormap (flipud (gray ()));

    % PMMH plot
    X = [];
    for j = 1:length (MCMC_FILES)
        nci = netcdf (MCMC_FILES{j}, 'r');
        localX = [];
        for i = 1:length (vars)
            x = read_var (nci, vars{i}, [], ps, 1);
            if logs (i)
                x = log(x);
            end
            localX = [ localX, x(:) ];
        end
        X = [ X; localX ];
    end
    Cor = cor(X,X);
    %Cor = chol(cov(X,X));
    
    
    %for i = 1:rows (Cor)
    %    for j = 1:columns (Cor)
    %        if abs(Cor(i,j)) < 1.0e-2
    %            Cor(i,j) = 0.0;
    %        end
    %    end
    %end
    
    subplot (1,2,1);
    [x y c] = hintmat (Cor);
    vertices = [ x'(:) y'(:) ];
    N = length (vertices);
    faces = reshape([1:N], 4, N/4)';
    patch ('Faces', faces, 'Vertices', vertices, 'FaceVertexCData', ...
        (c - 1).*rows (colormap));
    plot_defaults;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', [1:length(names)] - 0.5);
    set(gca, 'xticklabel', []);
    set(gca, 'ytick', [1:length(names)] - 0.5);
    set(gca, 'yticklabel', flipud(names));
    
    % URTSS plot
    subplot (1,2,2);
    nc = netcdf (URTS_FILE, 'r');
    Sigma = squeeze (nc{'smooth.Sigma'}(1,:,:));
    Sigma = tril(Sigma) + tril(Sigma, 1)';
    Sigma = Sigma(is,is);
    sd = sqrt (diag (Sigma));
    Cor = Sigma./(sd*sd');
    %Cor = chol(Sigma);
    
    %for i = 1:rows (Cor)
    %    for j = 1:columns (Cor)
    %        if abs(Cor(i,j)) < 1.0e-2
    %            Cor(i,j) = 0.0;
    %        end
    %    end
    %end
    
    [x y c] = hintmat (Cor);
    vertices = [ x'(:) y'(:) ];
    N = length (vertices);
    faces = reshape([1:N], 4, N/4)';
    patch ('Faces', faces, 'Vertices', vertices, 'FaceVertexCData', ...
        (c - 1).*rows (colormap));
    plot_defaults;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', [1:length(names)] - 0.5);
    set(gca, 'xticklabel', []);
    set(gca, 'ytick', [1:length(names)] - 0.5);
    set(gca, 'yticklabel', flipud(names));
end
