% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} krig ()
%
% Krig forcings.
% @end deftypefn
%
function model = krig (t, y, u, v)
    % initial values of mean hyperparameters
    a0 = (max(y) - min(y))/2; % initial amplitude
    phi0 = -pi; % initial phase
    c0 = mean(y); % initial intercept
    sf = a0/10.0; % variance amplitude
    sigma = sf/10.0;
    
    inffunc = @infExact;
    meanfunc = {@meanSum, {@meanSin, @meanConst}};
    hyp.mean = [a0; phi0; c0];
    covfunc = {@covSum, {@covSin, @covSEiso}};
    hyp.cov = log([a0; 0.1; a0; 30.0]);
    likfunc = @likGauss;
    hyp.lik = log(sigma);
                
    % train (@gpwrap instead of @gp removes optimisation of likelihood
    % hyperparameters)
    hyp = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, likfunc, t, y);
    
    % predict marginals
    [uymu uys2 uxmu uxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t, y, u);
    [vymu vys2 vxmu vxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t, y, v);

    % construct model
    model.inffunc = inffunc;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
    model.hyp = hyp;
    model.t = t;
    model.y = y;
    model.u = u;
    model.uxmu = uxmu;
    model.uxs2 = uxs2;
    model.v = v;
    model.vxmu = vxmu;
    model.vxs2 = vxs2;
end
