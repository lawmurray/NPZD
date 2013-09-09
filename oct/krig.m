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
function hyp = krig ()
    nc = netcdf('data/OSP_force_raw.nc', 'r');
    
    ts = nc{'time_FE'}(:);
    ys = log(nc{'FE'}(:))'; % log all but FT
    
    is = find(ts <= 3*365);
    js = find(ts > 3*365);

    t = ts(is)(:); % fit region
    y = ys(is)(:);
    
    u = [floor(ts(1)):ceil(ts(is(end)))]'; % prediction times in fit region
    v = [(u(end)+1):ceil(ts(end))]'; % prediction times in forecast region
    
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
    
    % predict
    [uymu uys2 uxmu uxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t, y, u);
    [vymu vys2 vxmu vxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t, y, v);
    
    clf;
    area_between(u, uxmu - 2*sqrt(uxs2), uxmu + 2*sqrt(uxs2), watercolour(1));
    hold on;
    area_between(v, vxmu - 2*sqrt(vxs2), vxmu + 2*sqrt(vxs2), watercolour(2));
    plot(u, uxmu, 'color', watercolour(1), 'linewidth', 3);
    plot(v, vxmu, 'color', watercolour(2), 'linewidth', 3);
    plot(ts, ys, 'ok');
    hold off;
end
