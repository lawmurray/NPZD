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
    nci = netcdf('results/simulate.nc.0', 'r');
    nco = netcdf('data/C10_OSP_71_76_likelihood_init.nc', 'c');    
    names = {
        'PgC';
        'PCh';
        'PRN';
        'ASN';
        'Zin';
        'ZCl';
        'ZgE';
        'Dre';
        'ZmQ';
        'EZ';
        'Chla';
        'P';
        'Z';
        'D';
        'N';
        'Kw';
        'KCh';
        'Dsi';
        'ZgD';
        'PDF';
        'ZDF';
        'muPgC';
        'muPCh';
        'muPRN';
        'muASN';
        'muZin';
        'muZCl';
        'muZgE';
        'muDre';
        'muZmQ';
    };
    
    % dimensions
    nco('np') = length(nci('np'));
    
    % variables
    for i = 1:length(names)
        x = squeeze(nci{names{i}}(end,:));
        nco{names{i}} = ncdouble('np');
        nco{names{i}}(:) = x;
    end
    
    ncclose(nci);
    ncclose(nco);
end

 
