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
  nyears = 4; % number of years for inference

  for i = 1:length(vars)
      ncvar = vars{i};
      time_ncvar = sprintf('time_%s', ncvar);
      
      ts = nc{time_ncvar}(:);
      ys = nc{ncvar}(:);
      if logs{i}
          ys = log(ys);
      end
      
      is = find(ts <= nyears*365);
      js = find(ts > nyears*365);
      
      t = ts(is)(:); % fit region
      y = ys(is)(:);
    
      u = [floor(ts(1)):ceil(ts(is(end)))]'; % prediction times in fit region
      v = [(u(end)+1):ceil(ts(end))]'; % prediction times in forecast region
      
      models{i} = krig(t, y, u, v);
      models{i} = sample(models{i}, 1);
      
  end
