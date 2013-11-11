function y = rho(t, theta, k)
  base = 4*(k - 1);
  alpha = theta(base + 1);
  psi = theta(base + 2);
  gamma = exp(theta(base + 3)) + 2;
  omega = theta(base + 4);
  
  y = alpha.*sin(2.*pi.*sigmoid(fmod(t .- psi .+ 365, 365)./730, gamma)) .+ omega;
end
