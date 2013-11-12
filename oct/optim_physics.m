% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>

% -*- texinfo -*-
% @deftypefn {Function File} optim_physics ()
%
% Least squares optimisation of mean function for physical variable.
% @end deftypefn
%
function [theta,ss,tt,xx] = optim_physics()
    nc = netcdf('data/obs_1d_osp.nc', 'r');
    t = nc{'time_T'}(:);
    s = nc{'coord_T'}(:);
    data = nc{'T'}(:);
    
    is = find(t <= 4*365);
    t = t(is);
    s = s(is);
    data = data(is);
    
    is = find(s <= 300);
    t = t(is);
    s = s(is);
    data = data(is);
    data = log(data);
    
    
    theta = [
        0.5; % alpha_1
        180; % psi_1
        2; % gamma_1
        2; % omega_1
        0; % alpha_2
        0; % ...
        1;
        0;
        1; % alpha_3
        0;
        1;
        3;
        1; % alpha_4
        0;
        1;
        3;
        10; % alpha_5
        0;
        1;
        20;
        40; % alpha_6
        0;
        1;
        80;
        1;  % alpha_7
        0;
        1; % ...
        200; % omega_7
        log(4); % kappa
        ];
    
    %checkgrad('cost', 1.0e-8, theta, s, t, data);
    theta = minimize(theta, @cost, -20, s, t, data);
    
    tt = [0:4*365]';
    ss = [1:300]';
    [tt,ss] = meshgrid(tt, ss);
    
    xx = mu(ss, tt, theta);
    
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
