function surf_physics_raster
    burnin = 0;
    nsamples = 256;
    
    nc = netcdf('results/filter.nc', 'r');
    %t = nc{'t'}(:,1);
    t = nc{'time'}(:);
    c = [0:699]';
    [tt,cc] = meshgrid(t, c);
    T = zeros(size(cc));

    for i = (burnin+1):nsamples
        theta(1:3) = nc{'alpha'}(:,i);
        theta(4:6) = nc{'psi'}(:,i);
        theta(7:9) = nc{'beta'}(:,i);
        theta(10) = nc{'a'}(i);
        theta(11) = nc{'c'}(i);
        
        x = nc{'x'}(:,:,i)';
 
        alpha = nc{'alpha'}(:,i);
        psi = nc{'psi'}(:,i);
        beta = nc{'beta'}(:,i);
        a = nc{'a'}(i);
        c = nc{'c'}(i);

        phi1 = alpha(1).*sin(2.*pi.*(tt .- psi(1))./365.0) .+ beta(1);
        phi2 = alpha(2).*sin(2.*pi.*(tt .- psi(2))./365.0) .+ beta(2);
        phi3 = alpha(3).*sin(2.*pi.*(tt .- psi(3))./365.0) .+ beta(3);

        %mu = phi1.*(0.5.*tanh((phi2 .- cc)./phi3) .+ 0.5) .+ a.*cc + c;
        %T += mu;
        
        T += mu(cc, tt, theta);
        T += x;
    end
    T /= nsamples - burnin;
    T = exp(T);
    ncclose(nc);
    
    % obs
    nc = netcdf('data/obs_1d_osp.nc', 'r');
    t = nc{'time_T'}(:);
    is = find(t <= 365);
    t = t(is);
    c = nc{'coord_T'}(is);
    y = nc{'T'}(is);
    ncclose(nc);
    
    s = surf(tt, -cc, T); shading('interp');
    hold on;
    scatter3(t, -c, y, 'k');
    h = stem3(t, -c, y, '-k');
    set(h, 'showbaseline', 'on');
    %set(h, 'baseline', s);
    %set(h, 'basevalue', mu(c, t, theta)); % needs to be mean surface
    set(h, 'markersize', 6);
    set(h, 'markerfacecolor', 'w');
    hold off;
end
