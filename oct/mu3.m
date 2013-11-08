function x = mu3(s, t, theta)
  kappa = theta(29);
  x = (kappa - rho(t, theta, 1).*sigmoid(0.5*tau(s, t, theta, 2), rho(t, theta, 4)) + rho(t, theta, 1);
end

