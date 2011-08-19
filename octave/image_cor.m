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
function image_cor ()
    MCMC_FILES = glob('results/mcmc_acupf-*.nc.*');
    URTS_FILE = 'results/urts.nc.0';
    ps = [20001:10:50000];
    vars = invars();
    logs = ones (30, 1);
    logs(3) = 0;

    names = vars;
    for i = 16:30
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

    subplot (1,2,1);
    %imagesc (abs(Cor));
    [x y c] = hintmat (Cor);
    vertices = [ x'(:) y'(:) ];
    N = length (vertices);
    faces = reshape([1:N], 4, N/4)';
    patch ('Faces', faces, 'Vertices', vertices, 'FaceVertexCData', ...
        (c - 1).*rows (colormap));
    plot_defaults;
    caxis ([0.0 1.0]);
    %colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', []);
    %set(gca, 'xticklabel', names);
    set(gca, 'ytick', 1:length(names));
    set(gca, 'yticklabel', names);
    
    % URTSS plot
    subplot (1,2,2);
    nc = netcdf (URTS_FILE, 'r');
    is = [25:39, 1:15];
    Sigma = squeeze (nc{'smooth.Sigma'}(1,:,:));
    Sigma = tril(Sigma) + tril(Sigma, 1)';
    Sigma = Sigma(is,is);
    sd = sqrt (diag (Sigma));
    Cor = Sigma./(sd*sd');
    %imagesc (abs(Cor));
    [x y c] = hintmat (Cor);
    vertices = [ x'(:) y'(:) ];
    N = length (vertices);
    faces = reshape([1:N], 4, N/4)';
    patch ('Faces', faces, 'Vertices', vertices, 'FaceVertexCData', ...
        (c - 1).*rows (colormap));
    plot_defaults;
    caxis ([0.0 1.0]);
    %colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', []);
    %set(gca, 'xticklabel', names);
    set(gca, 'ytick', 1:length(names));
    set(gca, 'yticklabel', names);
end
