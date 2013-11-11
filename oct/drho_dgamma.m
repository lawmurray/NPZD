function d = drho_dgamma(t, alpha, psi, gamma, omega)
  x = fmod(t .- psi .+ 365, 365)./730;
  a = gamma;

  df_dx = dsigmoid_dx(x, a);
  df_da = dsigmoid_da(x, a);

  dx_dgamma = 0;
  da_dgamma = 1;

  d = 2.*pi.*alpha.*(df_dx.*dx_dgamma .+ df_da.*da_dgamma).*cos(2.*pi.*sigmoid(x, a));
end
