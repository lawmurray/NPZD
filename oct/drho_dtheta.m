function d = drho_dtheta(t, theta, k)
  d = zeros(length(t), length(theta));
  base = 4*(k - 1);
  alpha = theta(base + 1);
  psi = theta(base + 2);
  gamma = theta(base + 3);
  omega = theta(base + 4);

  d_dalpha = drho_dalpha(t, alpha, psi, gamma, omega);
  d_dpsi = drho_dpsi(t, alpha, psi, gamma, omega);
  d_dgamma = drho_dgamma(t, alpha, psi, gamma, omega);
  d_domega = drho_domega(t, alpha, psi, gamma, omega);

  d(:,base + 1) = d_dalpha;
  d(:,base + 2) = d_dpsi;
  d(:,base + 3) = d_dgamma;
  d(:,base + 4) = d_domega;
end
