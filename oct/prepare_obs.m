function prepare_obs ()
    TRUTH_FILE = 'data/C10_TE_true_pad.nc';
    OBS_FILE = 'data/C10_TE_obs.nc';
    P = 1;
    
    nc = netcdf(OBS_FILE, 'c');
    ncclose(nc);

    gen_obs(TRUTH_FILE, 'N', OBS_FILE, 'N_obs', P, [366:730], 0.4, 1);
    gen_obs(TRUTH_FILE, 'Chla_lag', OBS_FILE, 'Chla_obs', P, [367:731], 0.4, ...
            1);
    
    nc = netcdf(OBS_FILE, 'w');
    nc{'time_N_obs'}(:) = nc{'time_N_obs'}(:) - 365.0;
    nc{'time_Chla_obs'}(:) = nc{'time_Chla_obs'}(:) - 366.0;
    ncclose(nc);
end
