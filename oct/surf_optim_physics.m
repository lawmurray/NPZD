function surf_optim_physics
    nc = netcdf('data/init_1d_osp.nc', 'r');
    theta(1:3) = nc{'alpha'}(:);
    theta(4:6) = nc{'psi'}(:);
    theta(7:9) = nc{'beta'}(:);
    theta(10) = nc{'a'}(:);
    theta(11) = nc{'c'}(:);
    ncclose(nc);
    
    t = [0:365]';
    s = [0:200]';
    [tt,ss] = meshgrid(t, s);
    x = mu(ss, tt, theta) ;
    
    % surface
    s = surf(tt, -ss, x);
    shading('interp');
    
    % obs
    nc = netcdf('data/obs_1d_osp.nc', 'r');
    t = nc{'time_T'}(:);
    s = nc{'coord_T'}(:);
    data = nc{'T'}(:);
    
    is = find(t <= 365);
    t = t(is);
    s = s(is);
    data = data(is);
    
    is = find(s <= 200);
    t = t(is);
    s = s(is);
    data = data(is);
    
    data = log(data);

    hold on;
    h = scatter3(t, -s, data, 'k');
    %set(h, 'showbaseline', 'on');
    %set(h, 'baseline', s);
    %set(h, 'basevalue', mu(c, t, theta)); % needs to be mean surface
    %set(h, 'markersize', 6);
    %set(h, 'markerfacecolor', 'w');
    hold off;
end
