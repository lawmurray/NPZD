% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} krig_biology ()
%
% Krig forcings.
% @end deftypefn
%
function model = krig_biology (t, y, u, sigma)    
    c0 = mean(y);
    
    inffunc = @infExact;
    meanfunc = @meanConst;
    hyp.mean = c0;
    covfunc = @covSEiso;
    hyp.cov = log([30.0; 1.0]);
    likfunc = @likGauss;
    hyp.lik = log(sigma);
                
    % train (@gpwrap instead of @gp removes optimisation of likelihood
    % hyperparameters)
    hyp = minimize(hyp, @gpwrap, -1000, @infExact, meanfunc, covfunc, likfunc, t, y);
    
    % predict marginals
    [ymu ys2 xmu xs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, t, y, u);

    % construct model
    model.inffunc = inffunc;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
    model.hyp = hyp;
    model.t = t;
    model.y = y;
    model.u = u;
    model.mu = xmu;
    model.s2 = xs2;
end
