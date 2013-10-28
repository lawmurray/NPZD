% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>

% -*- texinfo -*-
% @deftypefn {Function File} optim_physics ()
%
% Least squares optimisation of mean function for physical variable.
% @end deftypefn
%
function [theta,s,t,x] = optim_physics()
    nc = netcdf('data/obs_1d_osp.nc', 'r');
    s = nc{'coord_T'}(:);
    t = nc{'time_T'}(:);
    data = nc{'T'}(:);
    
    phi1 = (max(data) - min(data))/2;
    phi2 = max(s)/2;
    phi3 = max(s)/8;
    phi4 = phi1/2 + min(data);

    theta = [ 0 0 0 0 0 0 0 0 phi1 phi2 phi3 phi4 ]'; % alpha, psi, beta
    
    theta = minimize(theta, @cost, -5000, s, t, data);
    
    t = [0:7*365]';
    s = [1:700]';
    [t,s] = meshgrid(t, s);
    
    x = mu(s, t, theta);
end

function [val, dval] = cost(theta, s, t, data)
    val = f(s, t, data, theta);
    dval = df(s, t, data, theta);
end

function val = f(s, t, data, theta)
    val = meansq(data .- mu(s, t, theta));
end

function val = mu(s, t, theta)
    val = phi(t, 1, theta).*tanh((phi(t, 2, theta) .- s)./phi(t, 3, theta)) .+ phi(t, 4, theta);
end

function val = phi(t, k, theta)
    alpha = theta(1:4);
    psi = theta(5:8);
    beta = theta(9:12);
    val = alpha(k).*sin(2.*pi.*(t .- psi(k))./365) .+ beta(k);
end

function dval = df(s, t, data, theta)
    dval = -2.0.*mean((data .- mu(s, t, theta)).*dmu(s, t, theta));
end

function dval = dmu(s, t, theta)
    a1 = dphi(t, 1, theta).*tanh((phi(t, 2, theta) .- s)./phi(t, 3, theta));
    
    b1 = sech((phi(t, 2, theta) .- s)./phi(t, 3, theta)).**2;
    b2 = (dphi(t, 2, theta).*phi(t, 3, theta) .- dphi(t, 3, theta).*(phi(t, 2, theta) .- s))./(phi(t, 3, theta).**2);
    b3 = phi(t, 1, theta);
    a2 = b1.*b2.*b3;
    
    a3 = dphi(t, 4, theta);
    
    dval = a1 .+ a2 .+ a3;
end

function dval = dphi(t, k, theta)
    alpha = theta(k);
    psi = theta(4 + k);
    beta = theta(8 + k);
    
    dval = zeros(length(t), 12);
    
    dval(:,k) = sin(2.*pi.*(t .- psi)./365); % wrt alpha
    dval(:,4 + k) = alpha*cos(2*pi*(t - psi)/365)*(-2*pi/365); % wrt psi
    dval(:,8 + k) = 1; % wrt beta
end
