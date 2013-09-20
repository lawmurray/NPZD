% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} prepare_physics ()
%
% Krig forcings.
% @end deftypefn
%
function prepare_physics ()
  nc = netcdf('data/OSP_force_raw.nc', 'r');
  vars = {
    'FMLD';
    'BCN';
    'FT';
    'FE';
  };
  logs = {
    1;
    1;
    0;
    1;
  };
  sigmas = {
      0.2;
      0.1;
      0.05;
      0.05;
  };
  nyears = 4; % number of years for inference
  nsamples = 1000; % number of paths to sample
  u = [0:(7*365 - 1)]'; % prediction times

  % physics inference
  %load physics_models.mat
  for i = 1:length(vars)
      ncvar = vars{i};
      time_ncvar = sprintf('time_%s', ncvar);
      
      ts = nc{time_ncvar}(:);
      ys = nc{ncvar}(:);
      if logs{i}
          ys = log(ys);
      end
      
      is = find(ts < nyears*365);      
      t = ts(is)(:); % fit region
      y = ys(is)(:);
      
      models{i} = krig_physics(t, y, u, sigmas{i});
      models{i} = sample_physics(models{i}, nsamples);
      
      if logs{i}
          models{i}.X = exp(models{i}.X);
      end
  end
  ncclose(nc);
  save physics_models.mat models
  
  % derived and fixed forcings
  BCP = 0;
  BCZ = 0;
  BCD = 0;
  FX = 1;
  FMLD = models{1}.X;
  FMLC = [ zeros(nsamples, 1), diff(FMLD, 1, 2) ];
  FMIX = (1 + max(FMLC, 0))./FMLD;
  
  % create NetCDF file
  nc = netcdf('data/OSP_force.nc', 'c');
  nc('nr') = length(u);
  nc('np') = nsamples;
      
  nc{'time'} = ncdouble('nr');
  nc{'time'}(:) = u;
  
  for i = 1:length(vars)
      nc{vars{i}} = ncdouble('nr', 'np');
      nc{vars{i}}(:,:) = models{i}.X';
  end
    
  nc{'BCP'} = ncdouble();
  nc{'BCP'}(:) = BCP;
  nc{'BCZ'} = ncdouble();
  nc{'BCZ'}(:) = BCZ;
  nc{'BCD'} = ncdouble();
  nc{'BCD'}(:) = BCD;
  nc{'FX'} = ncdouble();
  nc{'FX'}(:) = FX;
  nc{'FMLC'} = ncdouble('nr', 'np');
  nc{'FMLC'}(:,:) = FMLC;
  nc{'FMIX'} = ncdouble('nr', 'np');
  nc{'FMIX'}(:,:) = FMIX;
  ncclose(nc);
end
