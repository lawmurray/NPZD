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
    MCMC_FILE = 'results/mcmc_pf-0.nc.0';
    URTS_FILE = 'results/urts.nc.0';
    ps = [25000:50000];
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
    nci = netcdf (MCMC_FILE, 'r');
    X = [];
    for i = 1:length (vars)
        x = read_var (nci, vars{i}, [], ps, 1);
        if logs (i)
            x = log(x);
        end
        X = [ X, x(:) ];
    end
    Cor = cor(X,X);

    subplot (1,2,1);
    imagesc (abs(Cor));
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
    Sigma = squeeze (nc{'smooth.Sigma'}(5,:,:));
    Sigma = tril(Sigma) + tril(Sigma, 1)';
    Sigma = Sigma(is,is);
    sd = sqrt (diag (Sigma));
    Cor = Sigma./(sd*sd');
    imagesc (abs(Cor));
    plot_defaults;
    caxis ([0.0 1.0]);
    %colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', []);
    %set(gca, 'xticklabel', names);
    set(gca, 'ytick', 1:length(names));
    set(gca, 'yticklabel', names);
end
