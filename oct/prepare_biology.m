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
      0.2;
      0.2;
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
end
