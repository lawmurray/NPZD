function d = drho_dpsi(t, alpha, psi, gamma, omega)
  x = fmod(t .- psi .+ 365, 365)./365;
  a = gamma;

  df_dx = dsigmoid_dx(x, a);
  df_da = dsigmoid_da(x, a);

  dx_dpsi = -1./365;
  da_dpsi = 0;

  d = 2.*pi.*alpha.*(df_dx.*dx_dpsi .+ df_da.*da_dpsi).*cos(2.*pi.*sigmoid(x, a));
end

