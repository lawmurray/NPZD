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
    ncwrite(init_file, 'Kw', 0.020219);
    ncwrite(init_file, 'KCh', 0.016772);
    ncwrite(init_file, 'Dsi', 5.6781);
    ncwrite(init_file, 'ZgD', 0.54331);
    ncwrite(init_file, 'PDF', 0.20884);
    ncwrite(init_file, 'ZDF', 0.09371);
    ncwrite(init_file, 'muPgC', 0.78802);
    ncwrite(init_file, 'muPCh', 0.042694);
    ncwrite(init_file, 'muPRN', 0.20349);
    ncwrite(init_file, 'muASN', 0.50564);
    ncwrite(init_file, 'muZin', 6.2798);
    ncwrite(init_file, 'muZCl', 0.15968);
    ncwrite(init_file, 'muZgE', 0.43887);
    ncwrite(init_file, 'muDre', 0.12111);
    ncwrite(init_file, 'muZmQ', 0.038316);
end
