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

% preprocessing
ml.nit(53) = [];
ml.nit_sml(53) = []; % get rid of outlier
ml.nit_time(53) = [];

% prepare obs time series
BCN_time = ml.nit_time;
BCN = ml.nit_sml.*14.0;

FT_time = ml.time;
FT = ml.t;

FE_time = [0:ny*365];
FE = S_par(1:length(FE_time)); % S_par runs from 1971 to 1978

FMLD_time = ml.time;
FMLD = ml.d;

%delta_N_obs_time = ml.nit_time;
%delta_N_obs = BCN - ml.nit.*14.0;

N_obs_time = ml.nit_time;
N_obs = ml.nit.*14.0;

Chla_obs_time = ml.chla_time;
Chla_obs = ml.chla;

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

%x1 = find(delta_N_obs_time >= syear);
%x2 = find(delta_N_obs_time <= fyear);
%x3 = intersect(x1, x2);
%delta_N_obs_time = (delta_N_obs_time(x3) - syear)*tscale;
%delta_N_obs = delta_N_obs(x3);

x1 = find(N_obs_time >= syear);
x2 = find(N_obs_time <= fyear);
x3 = intersect(x1, x2);
N_obs_time = (N_obs_time(x3) - syear)*tscale;
N_obs = N_obs(x3);

x1 = find(Chla_obs_time >= syear);
x2 = find(Chla_obs_time <= fyear);
x3 = intersect(x1, x2);
Chla_obs_time = (Chla_obs_time(x3) - syear)*tscale;
Chla_obs = Chla_obs(x3);

% Open NetCDF file (Octave version)
nc = netcdf('data/obs_osp.nc', 'c');

% Define the dimensions.
%nc('nr_BCN') = length(BCN);
%nc('nr_FT') = length(FT);
%nc('nr_FE') = length(FE);
%nc('nr_FMLD') = length(FMLD);
%nc('nr_delta_N_obs') = length(delta_N_obs);
nc('nr_N_obs') = length(N_obs);
nc('nr_Chla_obs') = length(Chla_obs);

% Define the variables
%nc{'time_BCN'} = ncdouble('nr_BCN');
%nc{'time_FT'} = ncdouble('nr_FT');
%nc{'time_FE'} = ncdouble('nr_FE');
%nc{'time_FMLD'} = ncdouble('nr_FMLD');
%nc{'time_delta_N_obs'} = ncdouble('nr_delta_N_obs');
nc{'time_N_obs'} = ncdouble('nr_N_obs');
nc{'time_Chla_obs'} = ncdouble('nr_Chla_obs');

%nc{'BCN'} = ncdouble('nr_BCN');
%nc{'FT'} = ncdouble('nr_FT');
%nc{'FE'} = ncdouble('nr_FE');
%nc{'FMLD'} = ncdouble('nr_FMLD');
%nc{'delta_N_obs'} = ncdouble('nr_delta_N_obs');
nc{'N_obs'} = ncdouble('nr_N_obs');
nc{'Chla_obs'} = ncdouble('nr_Chla_obs');

% Write data to variables
%nc{'time_BCN'}(:) = BCN_time;
%nc{'time_FT'}(:) = FT_time;
%nc{'time_FE'}(:) = FE_time;
%nc{'time_FMLD'}(:) = FMLD_time;
%nc{'time_delta_N_obs'}(:) = delta_N_obs_time;
nc{'time_N_obs'}(:) = N_obs_time;
nc{'time_Chla_obs'}(:) = Chla_obs_time;

%nc{'BCN'}(:) = BCN;
%nc{'FT'}(:) = FT;
%nc{'FE'}(:) = FE;
%nc{'FMLD'}(:) = FMLD;
%nc{'delta_N_obs'}(:) = delta_N_obs;
nc{'N_obs'}(:) = N_obs;
nc{'Chla_obs'}(:) = Chla_obs;

ncclose(nc)

end
