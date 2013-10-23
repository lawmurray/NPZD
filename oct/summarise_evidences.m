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
function summarise_evidences (suffix)
  if nargin < 1
    suffix = '';
  end
    
  le = zeros(1552,200);
  for i = [1 3:167 169:200]
      nc = netcdf(sprintf('results/posterior_smc2%s.nc.%d', suffix, i - 1), 'r');
      le(:,i) = nc{'logevidence'}(:);
      ncclose(nc);
  end
  le = sum(le);
  sigma = std(le([1 3:167 169:200]))
  ess = ess(le([1 3:167 169:200]))
end
