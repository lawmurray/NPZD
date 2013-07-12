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
    MCMC_FILES = glob('/tmp/mcmc_pf.nc.0');
    URTS_FILE = 'results/urts.nc.0';
    ps = [50001:1:100000];
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

    subplot (1,2,1);
    imagesc (abs(Cor));
    plot_defaults;
    caxis ([-1.0 1.0]);
    colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', [1:length(names)]);
    set(gca, 'xticklabel', []);
    set(gca, 'ytick', [1:length(names)]);
    set(gca, 'yticklabel', names);
    
    % URTSS plot
    subplot (1,2,2);
    nc = netcdf (URTS_FILE, 'r');
    Sigma = squeeze (nc{'smooth.Sigma'}(1,:,:));
    Sigma = tril(Sigma) + tril(Sigma, 1)';
    Sigma = Sigma(is,is);
    sd = sqrt (diag (Sigma));
    Cor = Sigma./(sd*sd');
    imagesc (abs(Cor));
    plot_defaults;
    caxis ([-1.0 1.0]);
    colorbar;
    set(gca, 'interpreter', 'tex');
    set(gca, 'xtick', 1:length(names));
    set(gca, 'xticklabel', []);
    set(gca, 'ytick', 1:length(names));
    set(gca, 'yticklabel', names);
end
