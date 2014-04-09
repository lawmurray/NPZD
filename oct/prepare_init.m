% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} prepare_init ()
%
% Prepare samples for likelihood program, from output of simulate program.
% @end deftypefn
%
function prepare_init ()
    init_file = 'data/init.nc';

    % create init file
    try
      nccreate(init_file, 'Kw');
      nccreate(init_file, 'KCh');
      nccreate(init_file, 'Dsi');
      nccreate(init_file, 'ZgD');
      nccreate(init_file, 'PDF');
      nccreate(init_file, 'ZDF');
      nccreate(init_file, 'muPgC');
      nccreate(init_file, 'muPCh');
      nccreate(init_file, 'muPRN');
      nccreate(init_file, 'muASN');
      nccreate(init_file, 'muZin');
      nccreate(init_file, 'muZCl');
      nccreate(init_file, 'muZgE');
      nccreate(init_file, 'muDre');
      nccreate(init_file, 'muZmQ');
    catch
        % assume variables already exist...
    end
    
    % initialise from previous pilot run
    ncwrite(init_file, 'Kw', 0.038883);
    ncwrite(init_file, 'KCh', 0.0083430);
    ncwrite(init_file, 'Dsi', 4.2340);
    ncwrite(init_file, 'ZgD', 0.44591);
    ncwrite(init_file, 'PDF', 0.29149);
    ncwrite(init_file, 'ZDF', 0.22972);
    ncwrite(init_file, 'muPgC', 0.47164);
    ncwrite(init_file, 'muPCh', 0.030454);
    ncwrite(init_file, 'muPRN', 0.22213);
    ncwrite(init_file, 'muASN', 1.0067);
    ncwrite(init_file, 'muZin', 3.9780);
    ncwrite(init_file, 'muZCl', 0.26615);
    ncwrite(init_file, 'muZgE', 0.29899);
    ncwrite(init_file, 'muDre', 0.13225);
    ncwrite(init_file, 'muZmQ', 0.0092010);
end
