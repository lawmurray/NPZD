% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} strip ()
%
% Strip down mcmc output files to bare essentials.
% @end deftypefn
%
function strip (dir)
    ins = glob(sprintf('%s/mcmc_*', dir));
    outs = strcat(ins, '.strip');
    vars = invars();
    
    for i = 1:length(ins)
        extract (ins{i}, outs{i}, vars, 1);
        movefile (outs{i}, ins{i}); % replace original with stripped file
    end
end
