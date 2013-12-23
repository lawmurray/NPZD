function y = rho(t, theta, k)
  % map k for eliminated sub-mixed layer
  map = [ 1; 0; 0; 2; 0; 3; 4 ];
  k = map(k);
  if k == 0
    error('trying to use sub-mixed layer');
  end

  base = 4*(k - 1);
  alpha = theta(base + 1);
  psi = theta(base + 2);
  gamma = theta(base + 3);
  omega = theta(base + 4);

  y = alpha.*sin(2.*pi.*sigmoid(fmod(t .- psi .+ 365, 365)./365, gamma)) .+ omega;
end
