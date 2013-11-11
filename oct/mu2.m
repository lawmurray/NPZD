function x = mu2(s, t, theta)
  x = rho(t, theta, 2).*bulge(tau(s, t, theta, 1), rho(t, theta, 3)) .+ rho(t, theta, 1);
end
