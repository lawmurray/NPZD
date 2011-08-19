TRUTH_FILE = 'data/C10_TE_true.nc';
TRUTH_PAD_FILE = 'data/C10_TE_true_pad.nc';
PAD = 365;
vars = invars();

nci = netcdf(TRUTH_PAD_FILE, 'r');
nco = netcdf(TRUTH_FILE, 'c');

nco('nr') = length(nci('nr')) - PAD;
nco('np') = length(nci('np'));
nco{'time'} = ncdouble('nr');
nco{'time'}(:) = nci{'time'}(PAD + 1:end) - PAD;

for i = 1:length (vars)
    if ncvarexists(nci, vars{i})
        nco{vars{i}} = ncdouble('nr', 'np');
        nco{vars{i}}(:,:) = nci{vars{i}}(PAD + 1:end,:);
    end
end
