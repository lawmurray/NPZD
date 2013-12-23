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
function prepare_biology ()
  nc = netcdf('data/obs_osp.nc', 'r');
  vars = {
    'N_obs';
    'Chla_obs';
  };
  logs = {
    1;
    1;
  };
  sigmas = {
      0.1;
      0.1;
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
      
      models{i} = krig_biology(t, y, u, sigmas{i});
  end
  ncclose(nc);
  save biology_models.mat models
  
  nc = netcdf('data/input_osp.nc', 'w');

  nc{'N_ell2'} = ncdouble();
  nc{'N_sf2'} = ncdouble();
  nc{'N_c'} = ncdouble();
  nc{'N_ell2'}(:) = exp(2*models{1}.hyp.cov(1));
  nc{'N_sf2'}(:) = exp(2*models{1}.hyp.cov(2));
  nc{'N_sf2'}(:) = models{1}.hyp.mean(1);

  nc{'Chla_ell2'} = ncdouble();
  nc{'Chla_sf2'} = ncdouble();
  nc{'Chla_c'} = ncdouble();
  nc{'Chla_ell2'}(:) = exp(2*models{2}.hyp.cov(1));
  nc{'Chla_sf2'}(:) = exp(2*models{2}.hyp.cov(2));
  nc{'Chla_sf2'}(:) = models{2}.hyp.mean(1);
 
  ncclose(nc);
end
