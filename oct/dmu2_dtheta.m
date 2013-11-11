function d = dmu2_dtheta(s, t, theta)
  d = zeros(length(s), length(theta));

  x = tau(s, t, theta, 1);
  a = rho(t, theta, 3);

  df_dx = dbulge_dx(x, a);
  df_da = dbulge_da(x, a);

  dx_dtheta = dtau_dtheta(s, t, theta, 1);
  da_dtheta = drho_dtheta(t, theta, 3);

  d = bulge(x, a).*drho_dtheta(t, theta, 2) .+ drho_dtheta(t, theta, 1) .+ (df_dx.*dx_dtheta .+ df_da.*da_dtheta).*rho(t, theta, 2);  
end
