function y = rho(t, theta, k)
  base = 4*(k - 1);
  alpha = theta(base + 1);
  psi = theta(base + 2);
  gamma = theta(base + 3);
  omega = theta(base + 4);

  y = alpha.*sin(2.*pi.*sigmoid(fmod(t .- psi .+ 365, 365)./365, gamma)) .+ omega;
end
