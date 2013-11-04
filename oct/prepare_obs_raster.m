% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>

% -*- texinfo -*-
% @deftypefn {Function File} prepare_obs
%
% Prepare NetCDF observation file.
% @end deftypefn
%
function prepare_obs_raster ()
    load data/chla.700
    load data/nit.700
    load data/tem.700
       
    nc = netcdf('data/obs_1d_raster_osp.nc', 'c');
    nc('ndepth') = 700;
    
    % year range
    syear = 71;
    fyear = 78;
    ny = fyear - syear;
    tscale = 365*24;
    
    % Chla
    time = chla(:,1);
    t1 = find(time >= syear);
    t2 = find(time <= fyear);
    is = intersect(t1, t2);
    time = (time(is) - syear)*tscale;    
    coord = chla(is,2) - 1; % 1m depth at coord 0
    raster = round(time)*700 + coord;    
    data = chla(is,3);
    
    nc('n_Chla') = length(raster);
    nc{'time_Chla'} = ncdouble('n_Chla');
    nc{'Chla'} = ncdouble('n_Chla');
    
    nc{'time_Chla'}(:) = raster;
    nc{'Chla'}(:) = data;

    % N
    time = nit(:,1);
    t1 = find(time >= syear);
    t2 = find(time <= fyear);
    is = intersect(t1, t2);
    time = (time(is) - syear)*tscale;
    coord = nit(is,2) - 1; % 1m depth at coord 0
    raster = round(time)*700 + coord;    
    data = nit(is,3).*14.0;
    
    nc('n_N') = length(raster);
    nc{'time_N'} = ncdouble('n_N');
    nc{'N'} = ncdouble('n_N');
    
    nc{'time_N'}(:) = raster;
    nc{'N'}(:) = data;

    % T
    time = tem(:,1);
    t1 = find(time >= syear);
    t2 = find(time <= fyear);
    is = intersect(t1, t2);
    time = (time(is) - syear)*tscale;
    coord = tem(is,2) - 1; % 1m depth at coord 0
    raster = round(time)*700 + coord;    
    data = tem(is,3);
    
    nc('n_T') = length(raster);
    nc{'time_T'} = ncdouble('n_T');
    nc{'T'} = ncdouble('n_T');
    
    nc{'time_T'}(:) = raster;
    nc{'T'}(:) = data;    
    
    ncclose(nc);
end
