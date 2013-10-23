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
function prepare_evidences ()
  le = zeros(1552,200);
  for i = 1:200
      nc = netcdf(sprintf('results/posterior_smc2.nc.%d', i - 1), 'r');
      le(:,i) = nc{'logevidence'}(:);
      ncclose(nc);
  end
  le = cumsum(le);
  save model_evidence.mat le
end
