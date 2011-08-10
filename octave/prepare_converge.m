% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} prepare_converge ()
%
% Compute $\hat{R}^p$ statistics (Brooks & Gelman 1998) for NPZD model runs.
%
% @end deftypefn
%
function prepare_converge ()
    experiments = {'pf', 'mupf', 'cupf', 'apf', 'amupf', 'acupf', 'pf-pmatch', ...
                  'mupf-pmatch', 'cupf', 'apf-pmatch', 'amupf-pmatch', 'acupf'};
    invar = invars();
    coord = [];
    rang = [21:20:50000];

    % construct arguments for parallel execution
    C = length(experiments);
    Rp = cell(C,1);
    ins = cell(C,1);
    invars = cell(C,1);
    coords = cell(C,1);
    rangs = cell(C,1);
    for i = 1:C
        ins{i} = glob(sprintf('results/mcmc_%s-[0-9]*.nc.*', experiments{i}));
        invars{i} = invar;
        coords{i} = coord;
        rangs{i} = rang;
    end
   
    % execute
    Rp = parcellfun(C, @converge, ins, invars, coords, rangs, ...
        'UniformOutput', 0);
    
    % save
    save -binary Rp.mat Rp
end
