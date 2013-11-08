function d = drho_dtheta(t, theta, k)
  d = zeros(length(t), length(theta));
  base = 4*(k - 1);
  [alpha; psi; gamma; omega] = theta((base+1):(base+4));

  d_dalpha = drho_dalpha(t, theta, alpha, psi, gamma, omega);
  d_dpsi = drho_dpsi(t, theta, alpha, psi, gamma, omega);
  d_dgamma = drho_dgamma(t, theta, alpha, psi, gamma, omega);
  d_domega = drho_domega(t, theta, alpha, psi, gamma, omega);
  
  d((base+1):(base+4)) = [d_dalpha; d_dpsi; d_dgamma; d_domega];
end

