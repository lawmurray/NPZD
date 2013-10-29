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
    
    phi1 = 2*(max(data) - min(data)); % for T, more like decay, no plateau
    phi2 = phi1/4;
    phi3 = 700/8;
    a = 0;
    c = min(data);

    theta = [ 0 0 0 0 0 0 phi1 phi2 phi3 a c ]'; % alpha, psi, beta
    
    theta = minimize(theta, @cost, -10000, s, t, data);
    
    t = [0:7*365]';
    s = [1:700]';
    [t,s] = meshgrid(t, s);
    
    x = mu(s, t, theta);
    
    nc = netcdf('data/init_1d_osp.nc', 'c');
    nc('z') = 700;
    nc('n') = 3;
    
    nc{'alpha'} = ncdouble('n');
    nc{'psi'} = ncdouble('n');
    nc{'beta'} = ncdouble('n');
    nc{'a'} = ncdouble();
    nc{'c'} = ncdouble();
    nc{'coord_T'} = ncint('z');
    nc{'T'} = ncdouble('z');
    
    nc{'alpha'}(:) = theta(1:3);
    nc{'psi'}(:) = theta(4:6);
    nc{'beta'}(:) = theta(7:9);
    nc{'a'}(:) = theta(10);
    nc{'c'}(:) = theta(11);
    nc{'coord_T'}(:) = [0:699]';
    nc{'T'}(:) = x(:,1);
    ncclose(nc);
end

function [val, dval] = cost(theta, s, t, data)
    val = f(s, t, data, theta);
    dval = df(s, t, data, theta);
end

function val = f(s, t, data, theta)
    val = meansq(data .- mu(s, t, theta));
end

function val = mu(s, t, theta)
    val = phi(t, 1, theta).*(0.5*tanh((phi(t, 2, theta) .- s)./phi(t, 3, theta)) .+ 0.5) .+ lin(s, theta);
end

function val = phi(t, k, theta)
    alpha = theta(1:3);
    psi = theta(4:6);
    beta = theta(7:9);
    val = alpha(k).*sin(2.*pi.*(t .- psi(k))./365) .+ beta(k);
end

function val = lin(s, theta)
   a = theta(10);
   c = theta(11);
   val = a.*s .+ c;
end

function dval = df(s, t, data, theta)
    dval = -2.0.*mean((data .- mu(s, t, theta)).*dmu(s, t, theta));
end

function dval = dmu(s, t, theta)
    a1 = 0.5*dphi(t, 1, theta).*tanh((phi(t, 2, theta) .- s)./phi(t, 3, theta));
    a2 = 0.5*dphi(t, 1, theta);
    
    b1 = sech((phi(t, 2, theta) .- s)./phi(t, 3, theta)).**2;
    b2 = (dphi(t, 2, theta).*phi(t, 3, theta) .- dphi(t, 3, theta).*(phi(t, 2, theta) .- s))./(phi(t, 3, theta).**2);
    b3 = 0.5*phi(t, 1, theta);
    a3 = b1.*b2.*b3;
    
    a4 = dlin(s, theta);
    
    dval = a1 .+ a2 .+ a3 .+ a4;
end

function dval = dphi(t, k, theta)
    alpha = theta(k);
    psi = theta(4 + k);
    beta = theta(8 + k);
    
    dval = zeros(length(t), 11);
    
    dval(:,k) = sin(2.*pi.*(t .- psi)./365); % wrt alpha
    dval(:,3 + k) = alpha*cos(2*pi*(t - psi)/365)*(-2*pi/365); % wrt psi
    dval(:,6 + k) = 1; % wrt beta
end

function dval = dlin(s, theta)
    dval = zeros(length(s), 11);
    dval(:,10) = s;    
    dval(:,11) = 1;
end
