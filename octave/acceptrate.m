% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} acceptrate (ex)
%
% Compute acceptance rate.
% @end deftypefn
%
function [mu sigma] = acceptrate (ex)
    if nargin != 1
        print_usage ();
    end
    
    files = glob(sprintf('results/mcmc_%s-[0-9]*.nc.*', ex));
    N = length(files);
    as = zeros(N,1);
    for i = 1:N
        nc = netcdf(files{i}, 'r');
        ll = nc{'loglikelihood'}(:);
        as(i) = length(unique(ll))/length(ll);
    end
    
    mu = mean(as);
    sigma = std(as);
end
