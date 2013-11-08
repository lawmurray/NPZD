function d = dmu3_dtheta(s, t, theta)
  kappa = theta(29);
  d = zeros(length(s), length(theta));

  x = 0.5*tau(s, t, theta, 2);
  a = rho(t, theta, 4);

  df_dx = dsigmoid_dx(x, a);
  df_da = dsigmoid_da(x, a);

  dx_dtheta = 0.5*dtau_dtheta(s, t, theta, 2);
  da_dtheta = drho_dtheta(t, theta, 4);

  d = -sigmoid(x, a).*drho_dtheta(t, theta, 1) .+ drho_dtheta(t, theta, 1) .+ (df_dx.*dx_dtheta .+ df_da.*da_dtheta).*(kappa .- rho(t, theta, 1);
end

