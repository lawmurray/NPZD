function x = mu3(s, t, theta)
  kappa = theta(end);
  x = (kappa .- rho(t, theta, 1)).*sigmoid(tau(s, t, theta, 2), rho(t, theta, 4)) .+ rho(t, theta, 1);
end
