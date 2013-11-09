function d = dtau_dtheta(s, t, theta, k)
  mx = rho(t, theta, 5 + k);
  mn = rho(t, theta, 4 + k);
  dmx_dtheta = drho_dtheta(t, theta, 5 + k);
  dmn_dtheta = drho_dtheta(t, theta, 4 + k);

  d = -(s .- mn).*(dmx_dtheta .- dmn_dtheta)./((mx .- mn).^2) .- dmn_dtheta./(mx .- mn);
  d(find(mx .- mn == 0)) = 0;
end
