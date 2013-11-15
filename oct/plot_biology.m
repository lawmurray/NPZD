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
function plot_biology ()
  load biology_models.mat

  vars = {
    'N_obs';
    'Chla_obs';
  };
  logs = {
      1;
      1;
  };
  display_names = {
    'N_obs';
    'Chla_obs';
  };
  nyears = 4;
  t = [1:nyears*365]';

  clf;
  for i = 1:length(vars)
    m = models{i};
    subplot(2, 1, i);

    if logs{i}
        upper = logninv(0.975, m.mu, sqrt(m.s2));
        lower = logninv(0.025, m.mu, sqrt(m.s2));
        median = logninv(0.5, m.mu, sqrt(m.s2));
    else
        upper = norminv(0.975, m.mu, sqrt(m.s2));
        lower = norminv(0.025, m.mu, sqrt(m.s2));
        median = norminv(0.5, m.mu, sqrt(m.s2));
    end
    
    area_between(t, lower(t), upper(t), watercolour(1), 1.0, 0.3);
    hold on;
    plot(t, median(t), 'color', watercolour(1), 'linewidth', 3);
        
    nc = netcdf('data/obs_osp.nc', 'r');
    style = get_style([], [], 'data/obs_osp.nc', vars{i});
    t1 = bi_read_times(nc, vars{i});
    y1 = bi_read_var(nc, vars{i});
    is = find(t1 < nyears*365);
    t1 = t1(is);
    y1 = y1(is);
    plot (t1, y1, struct2options (style){:});
    hold off;
    
    grid on;
    axis('tight');
    ylabel(display_names{i});
    if i == 2
        xlabel('t');
    end
  end
end
