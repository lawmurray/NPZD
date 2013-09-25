% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} krig_physics ()
%
% Krig forcings.
% @end deftypefn
%
function model = krig_physics (t, y, u, sigma)
    % initial values of mean hyperparameters
    a0 = (max(y) - min(y))/2; % initial amplitude
    phi0 = -pi; % initial phase
    c0 = mean(y); % initial intercept
    sf = a0/10.0; % variance amplitude
    
    inffunc = @infExact;
    meanfunc = {@meanSum, {@meanSin, @meanConst}};
    hyp.mean = [a0; phi0; c0];
    covfunc = {@covSum, {@covSin, @covSEiso}};
    hyp.cov = log([1.0; sf; 30.0; sf]);
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
