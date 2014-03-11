% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
function prepare_input ()
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

  % prepare weight function parameters
  ell2 = zeros(length(vars), 1);
  sf2 = zeros(length(vars), 1);
  c = zeros(length(vars), 1);
  
  for i = 1:length(vars)
      ncvar = vars{i};
      time_ncvar = sprintf('time_%s', ncvar);
      
      ts = nc{time_ncvar}(:);
      ys = nc{ncvar}(:);
      if logs{i}
          ys = log(ys);
      end
      
      is = find(ts < nyears*365);      
      t = ts(is)(:);
      y = ys(is)(:);
      
      inffunc = @infExact;
      meanfunc = @meanConst;
      hyp.mean = mean(y);
      covfunc = @covSEiso;
      hyp.cov = log([30.0; 1.0]);
      likfunc = @likGauss;
      hyp.lik = log(sigmas{i});

      hyp = minimize(hyp, @gpwrap, -1000, @infExact, meanfunc, covfunc, likfunc, t, y);

      ell2(i) = exp(2*hyp.cov(1));
      sf2(i) = exp(2*hyp.cov(2));
      c(i) = hyp.mean(1);
  end
  ncclose(nc);

  % write weight function parameters to input file
  nc = netcdf('data/input_osp.nc', 'w');
  nc{'N_ell2'}(:) = ell2(1);
  nc{'N_sf2'}(:) = sf2(1);
  nc{'N_c'}(:) = c(1);;
  nc{'Chla_ell2'}(:) = ell2(2);
  nc{'Chla_sf2'}(:) = sf2(2);
  nc{'Chla_c'}(:) = c(2); 
  ncclose(nc);
end
