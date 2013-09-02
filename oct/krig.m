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
function krig ()
    nc = netcdf('data/C10_OSP_71_76_force.nc', 'r');
    
    t = nc{'time'}(:)(:);
    FT = nc{'FT'}(:)';
    FE = nc{'FE'}(:)';
    y = FE;
    
    T = 800;
    u = linspace(t(1), t(T), 1000)';
    v = linspace(t(T), t(end), 1000)';
    
    inffunc = @infExact;
    meanfunc = {@meanSum, {@meanSin, @meanLinear, @meanConst}};
    hyp.mean = [10; 91; log(365); 0.0; mean(y)]; % 91 for FE, 182 for FT
    covfunc = {@covSum, {@covPeriodic, @covSEiso}};
    hyp.cov = log([10; 365/2; 1; 30; 10]);
    likfunc = @likGauss;
    hyp.lik = log(0.1);
                
    % train (gpwrap removes optimisation of likelihood hyperparameters)
    hyp = minimize(hyp, @gpwrap, -500, @infExact, meanfunc, covfunc, likfunc, t(1:T), y(1:T));
    
    % predict
    [uymu uys2 uxmu uxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t(1:T), y(1:T), u);
    [vymu vys2 vxmu vxs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t(1:T), y(1:T), v);
    
    clf;
    area_between(u, uxmu - 2*sqrt(uxs2), uxmu + 2*sqrt(uxs2), watercolour(1));
    hold on;
    area_between(v, vxmu - 2*sqrt(vxs2), vxmu + 2*sqrt(vxs2), watercolour(2));
    plot(u, uxmu, 'color', watercolour(1), 'linewidth', 3);
    plot(v, vxmu, 'color', watercolour(2), 'linewidth', 3);
    plot(t, y, 'ok');
    hold off;
end
