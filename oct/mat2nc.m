function mat2nc

% load observational data
load data/OSP_ML_OBS.mat
load data/OSP_TS_ml_td.mat
load data/smooth_mld2.mat
load data/PAR_71_78.mat
load data/mlt.mat

% year range
syear = 71;
fyear = 78;
%% ^^^ Is there data for FE (S_par) beyond 1978?

ny = fyear - syear;
tscale = 365;
%% ^^^ Do we have to worry about leap years here?

% Some preprocessing
ml.nit_sml(53)=[]; %get rid of the "bad" data
ml.nit_time(53)=[];
%% ^^^ Is this just an outlier?

% Prepare raw forcings
BCN_time = ml.nit_time;
BCN = ml.nit_sml.*14.0;

FT_time = ml.time;
FT = ml.t;

FE_time = [0:ny*365];
FE = S_par(1:length(FE_time));
%% ^^^ I gather S_par runs from 1971 to 1978?

FMLD_time = ml.time;
FMLD = ml.d;
%% ^^^ Original file uses FMLD = mld_WMA, but this is smoothed already, yes?

% Restrict everything to the right year ranges (FE is okay already)
x1 = find(BCN_time >= syear);
x2 = find(BCN_time <= fyear);
x3 = intersect(x1, x2);
BCN_time = (BCN_time(x3) - syear)*tscale;
BCN = BCN(x3);

x1 = find(FT_time >= syear);
x2 = find(FT_time <= fyear);
x3 = intersect(x1, x2);
FT_time = (FT_time(x3) - syear)*tscale;
FT = FT(x3);

x1 = find(FMLD_time >= syear);
x2 = find(FMLD_time <= fyear);
x3 = intersect(x1, x2);
FMLD_time = (FMLD_time(x3) - syear)*tscale;
FMLD = FMLD(x3);

% Open NetCDF file (Octave version)
ncid_force = netcdf('data/OSP_force_raw.nc', 'c');

% Define the dimensions.
ncid_force('nr_BCN') = length(BCN);
ncid_force('nr_FT') = length(FT);
ncid_force('nr_FE') = length(FE);
ncid_force('nr_FMLD') = length(FMLD);

% Define the variables
ncid_force{'time_BCN'} = ncdouble('nr_BCN');
ncid_force{'time_FT'} = ncdouble('nr_FT');
ncid_force{'time_FE'} = ncdouble('nr_FE');
ncid_force{'time_FMLD'} = ncdouble('nr_FMLD');

ncid_force{'BCN'} = ncdouble('nr_BCN');
ncid_force{'FT'} = ncdouble('nr_FT');
ncid_force{'FE'} = ncdouble('nr_FE');
ncid_force{'FMLD'} = ncdouble('nr_FMLD');

% Write data to variables
ncid_force{'time_BCN'}(:) = BCN_time;
ncid_force{'time_FT'}(:) = FT_time;
ncid_force{'time_FE'}(:) = FE_time;
ncid_force{'time_FMLD'}(:) = FMLD_time;

ncid_force{'BCN'}(:) = BCN;
ncid_force{'FT'}(:) = FT;
ncid_force{'FE'}(:) = FE;
ncid_force{'FMLD'}(:) = FMLD;

ncclose(ncid_force)

end
