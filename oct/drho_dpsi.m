function d = drho_dpsi(t, alpha, psi, gamma, omega)
  x = (t .- psi)./730;
  a = gamma;

  df_dx = dsigmoid_dx(x, a);
  df_da = dsigmoid_da(x, a);

  dx_dtheta = -1/730;
  da_dtheta = 0;

  d = 2*pi*alpha*(df_dx.*dx_dtheta + df_da.*da_dtheta)*cos(2*pi*sigmoid(x, a));
end

