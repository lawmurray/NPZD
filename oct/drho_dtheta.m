function d = drho_dtheta(t, theta, k)
  d = zeros(length(t), length(theta));
  base = 4*(k - 1);
  [alpha; psi; gamma; omega] = theta((base+1):(base+4));

  d_dalpha = drho_dalpha(t, alpha, psi, gamma, omega);
  d_dpsi = drho_dpsi(t, alpha, psi, gamma, omega);
  d_dgamma = drho_dgamma(t, alpha, psi, gamma, omega);
  d_domega = drho_domega(t, alpha, psi, gamma, omega);
  
  d(:,(base+1):(base+4)) = [d_dalpha; d_dpsi; d_dgamma; d_domega];
end
