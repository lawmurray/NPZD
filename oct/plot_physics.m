% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_physics (model)
%
% Plot forcings from Gaussian process model.
% @end deftypefn
%
function plot_physics (m)
      clf;
      area_between(m.u, m.uxmu - 2*sqrt(m.uxs2), m.uxmu + 2*sqrt(m.uxs2), watercolour(1));
      hold on;
      area_between(m.v, m.vxmu - 2*sqrt(m.vxs2), m.vxmu + 2*sqrt(m.vxs2), watercolour(2));
      plot(m.u, m.uxmu, 'color', watercolour(1), 'linewidth', 3);
      plot(m.v, m.vxmu, 'color', watercolour(2), 'linewidth', 3);
      plot(m.t, m.y, 'ok');
      hold off;
end
