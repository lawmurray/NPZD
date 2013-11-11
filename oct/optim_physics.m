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
    t = nc{'time_T'}(:);
    s = nc{'coord_T'}(:);
    data = nc{'T'}(:);
    
    is = find(t <= 365);
    t = t(is);
    s = s(is);
    data = data(is);
    
    is = find(s <= 300);
    t = t(is);
    s = s(is);
    data = data(is);
    data = log(data);
    
    
    theta = [
        log(4); % alpha_1
        180; % psi_1
        log(0.0001); % gamma_1
        log(10); % omega_1
        0; % alpha_2
        0; % ...
        log(4);
        0;
        1; % alpha_3
        0;
        log(4);
        3;
        1; % alpha_4
        0;
        log(4);
        3;
        40; % alpha_5
        0;
        log(4);
        80;
        40; % alpha_6
        0;
        log(4);
        80;
        1;  % alpha_7
        0;
        log(4); % ...
        200; % omega_7
        log(4); % kappa
        ];
    
    %checkgrad('cost', 1.0e-8, theta, s, t, data);
    theta = minimize(theta, @cost, -2000, s, t, data);
    
    t = [0:365]';
    s = [1:300]';
    [t,s] = meshgrid(t, s);
    
    x = mu(s, t, theta);
    
    %nc = netcdf('data/init_1d_osp.nc', 'c');
    %nc('z') = 700;
    %nc('n') = 3;
    
    %nc{'alpha'} = ncdouble('n');
    %nc{'psi'} = ncdouble('n');
    %nc{'beta'} = ncdouble('n');
    %nc{'a'} = ncdouble();
    %nc{'c'} = ncdouble();
    %nc{'coord_T'} = ncint('z');
    %nc{'T'} = ncdouble('z');
    
    %nc{'alpha'}(:) = theta(1:3);
    %nc{'psi'}(:) = theta(4:6);
    %nc{'beta'}(:) = theta(7:9);
    %nc{'a'}(:) = theta(10);
    %nc{'c'}(:) = theta(11);
    %nc{'coord_T'}(:) = [0:699]';
    %nc{'T'}(:) = x(:,1);
    %ncclose(nc);
end
