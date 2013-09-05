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
    
    ts = nc{'time_FT'}(:);
    ys = nc{'FT'}(:)';    
    
    is = find(ts <= 5*365);
    js = find(ts > 5*365);

    t = ts(is)(:); % fit region
    y = ys(is)(:);
    
    u = [floor(ts(1)):ceil(ts(is(end)))]'; % prediction times in fit region
    v = [(u(end)+1):ceil(ts(end))]'; % prediction times in forecast region
    
    % initial values of mean hyperparameters
    a0 = (max(y) - min(y))/2; % initial amplitude
    phi0 = -pi; % initial phase
    T0 = log(365); % initial period
    b0 = (y(end) - y(1))/(t(end) - t(1)); % initial drift
    c0 = mean(y); % initial intercept
    
    inffunc = @infExact;
    meanfunc = {@meanSum, {@meanSin, @meanLinear, @meanConst}};
    hyp.mean = [a0; phi0; T0; b0; c0];
    covfunc = {@covSum, {@covPeriodic, @covSEiso}};
    hyp.cov = log([10; 365/2; 1; 30; 10]);
    likfunc = @likGauss;
    hyp.lik = log(0.1);
                
    % train (@gpwrap instead of @gp removes optimisation of likelihood
    % hyperparameters)
    hyp = minimize(hyp, @gp, -1000, @infExact, meanfunc, covfunc, likfunc, t, y);
    
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
