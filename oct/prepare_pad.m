FORCE_FILE = 'data/C10_TE_force.nc';
FORCE_PAD_FILE = 'data/C10_TE_force_pad.nc';

vars = {
    'BCN';
    'BCP';
    'BCZ';
    'BCD';
    'FX';
    'FT';
    'FE';
    'FMLD';
    'FMLC';
    'FMIX';
};

nci = netcdf(FORCE_FILE, 'r');
nco = netcdf(FORCE_PAD_FILE, 'c');

nco('nr') = 2*length(nci('nr'));
nco('ns') = length(nci('ns'));
nco{'time'} = ncdouble('nr');
nco{'time'}(:) = [0:(length(nco('nr')) - 1)];

for i = 1:length (vars)
    nco{vars{i}} = ncdouble('nr');
    x = nci{vars{i}}(:);
    x = [ x x ];
    nco{vars{i}}(:) = x;
end
