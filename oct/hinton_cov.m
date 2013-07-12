% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} hinton_cov (@var{type})
%
% Produce plot of posterior covariance for NPZD model.
%
% @itemize
% @bullet{ @var{type} 1 for URTSS initial conditions, 2 for URTSS
% parameters, 3 for UKF parameters.}
% @end itemize
% @end deftypefn
%
function hinton_cov (type)
    if nargin < 1
        type = 1;
    end
    
    MCMC_FILES = glob('results/mcmc_acupf-0.nc.*');
    ps = [50000:75000];
 
    if type == 1
        APPROX_FILE = 'results/urts.nc.0';
        var = 'smooth.Sigma';
        t = 1;
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
            %'EZ';
            'Chla_lag';
            'P';
            'Z';
            'D';
            'N';
        };
        names = vars;
        for i = 1:length (vars)
            names{i} = strcat ('{', vars{i}, '_0}');
        end
        is = [1:9 11:15];
        logs = ones(15,1);
    else
        if type == 2
            APPROX_FILE = 'results/urts.nc.0';
            var = 'smooth.Sigma';
            t = 1;
        else
            APPROX_FILE = 'results/ukf.nc.0';
            var = 'filter.Sigma';
            t = 101;
        end
        vars = {
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
        names = cell(length(vars), 1);
        for i = 1:length(vars)
            names{i} = papernames(){i};
        end
        is = [25:39];
        logs = ones(15,1);
        logs(3) = 0;
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
    Sigma = cov(X,X);
    
    subplot (1,2,1);
    [x y c] = hintmat (Sigma);
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
    
    % proposal plot
    nc = netcdf (APPROX_FILE, 'r');
    Sigma = squeeze (nc{var}(t,:,:));
    Sigma = tril(Sigma) + tril(Sigma, 1)';
    Sigma = Sigma(is,is);
    
    subplot (1,2,2);
    [x y c] = hintmat (Sigma);
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
