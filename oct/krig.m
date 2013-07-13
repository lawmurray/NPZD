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
    
    inffunc = @infExact;
    meanfunc = @meanSin; hyp.mean = [20; 10; 90; log(365)];
    %meanfunc = @meanConst; hyp.mean = [1];
    %covfunc = @covPeriodic; hyp.cov = log([1; 365/2; 1]);
    %covfunc = @covSEiso; hyp.cov = log([30; sqrt(100)]);
    covfunc = {@covSum, {@covPeriodic, @covSEiso}}; hyp.cov = [ log([10; ...
                        365/2; 1]); log([30; 10])];
    likfunc = @likGauss; hyp.lik = log(1.0);
    
    % train
    hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, t(1:T), y(1:T))
    %[nlZ dnlZ] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, t, y);
    
    % predict
    [ymu ys2 xmu xs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
                           t(1:T), y(1:T), t);
    
    clf;
    area_between(t(1:T), xmu(1:T) - 2*sqrt(xs2(1:T)), xmu(1:T) + 2* ...
                 sqrt(xs2(1:T)), watercolour(1));
    hold on;
    area_between(t((T+1):end), xmu((T+1):end) - 2*sqrt(xs2((T+1):end)), ...
                 xmu((T+1):end) + 2*sqrt(xs2((T+1):end)), watercolour(2));
    plot(t(1:T), xmu(1:T), 'color', watercolour(1), 'linewidth', 3);
    plot(t((T+1):end), xmu((T+1):end), 'color', watercolour(2), 'linewidth', 3);
    plot(t, y, 'ok');
    hold off;
end
