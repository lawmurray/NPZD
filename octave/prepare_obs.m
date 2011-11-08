function prepare_obs ()
    prepare_nonpad();
    TRUTH_FILE = 'data/C10_TE_true.nc';
    OBS_FILE = 'data/C10_TE_obs.nc';
    P = 2;
    
    gen_obs(TRUTH_FILE, 'N', OBS_FILE, 'N_obs', P, [], 0.2, 1);
    gen_obs(TRUTH_FILE, 'Chla', OBS_FILE, 'Chla_obs', P, [], 0.2, 1);
end
