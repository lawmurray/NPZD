% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_physics ()
%
% Plot forcings from Gaussian process model.
% @end deftypefn
%
function plot_physics ()
  load physics_models.mat
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
  yax = {
      [0 300];
      [0 500];
      [3 8];
      [0 50];
  };
  display_names = {
    'MLD';
    'BCN';
    'T';
    'E';
  };
  nyears = 4; % number of years for inference
  t = [1:nyears*365]';
  u = [nyears*365:7*365]';

  clf;
  for i = 1:length(vars)
    m = models{i};
    subplot(4, 1, i);

    if logs{i}
        upper = logninv(0.975, m.mu, sqrt(m.s2));
        lower = logninv(0.025, m.mu, sqrt(m.s2));
        median = logninv(0.5, m.mu, sqrt(m.s2));
    else
        upper = norminv(0.975, m.mu, sqrt(m.s2));
        lower = norminv(0.025, m.mu, sqrt(m.s2));
        median = norminv(0.5, m.mu, sqrt(m.s2));
    end
    
    %bi_plot_quantiles('data/OSP_force.nc', vars{i}, [], [], t, 1);
    area_between(t, lower(t), upper(t), watercolour(1), 1.0, 0.3);
    hold on;
    plot(t, median(t), 'color', watercolour(1), 'linewidth', 3);
    
    %bi_plot_quantiles('data/OSP_force.nc', vars{i}, [], [], u, 2);
    area_between(u, lower(u), upper(u), watercolour(2), 1.0, 0.3);
    plot(u, median(u), 'color', watercolour(2), 'linewidth', 3);
    
    style = get_style([], [], 'data/OSP_force_raw.nc', vars{i});
    t1 = bi_read_times(nc, vars{i});
    y1 = bi_read_var(nc, vars{i});
    %if logs{i}
    %    y1 = log(y1);
    %end
    plot (t1, y1, struct2options (style){:});
    hold off;
    
    grid on
    ax = axis();
    axis([t(1) u(end) yax{i}]);
    ylabel(display_names{i});
    if i == 4
        xlabel('t');
    end
  end
end
