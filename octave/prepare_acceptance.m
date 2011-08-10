% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} prepare_acceptance ()
%
% Prepare acceptance rates for NPZD model.
% @end deftypefn
%
function prepare_acceptance ()
    experiments = {'pf', 'mupf', 'cupf', 'apf', 'mupf', 'acupf', ...
        'pf-pmatch', 'mupf-pmatch', 'cupf', 'apf-pmatch', 'mupf-pmatch', ...
        'acupf'};
    invar = invars();
    M = 200;
    iter = 2000;
    attempt = 50;
    logvar = [1:2 4:30];
    
    % construct arguments for parallel execution
    C = length(experiments);
    files = cell(C,1);
    invars = cell(C,1);
    Ms = cell(C,1);
    coords = cell(C,1);
    attempts = cell(C,1);
    iters = cell(C,1);
    logvars = cell(C,1);
    for i = 1:length(experiments)
        files{i} = glob(sprintf('results/likelihood_%s-[0-9]*.nc.*', ...
            experiments{i}));
        invars{i} = invar;
        coords{i} = [];
        Ms{i} = M;
        iters{i} = iter;
        attempts{i} = attempt;
        logvars{i} = logvar;
    end

    % construct and krig models
    models = cellfun(@model_acceptance, files, invars, coords, Ms, ...
        logvars, 'UniformOutput', 0);
    models = cellfun(@krig_model, models, iters, 'UniformOutput', 0);
    mns = cellfun(@min_model, models, attempts, iters, 'UniformOutput', 0);
    mxs = cellfun(@max_model, models, attempts, iters, 'UniformOutput', 0);
    
    % save
    save -binary model_acceptance.mat models mns mxs
end
