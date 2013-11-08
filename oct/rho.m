function y = rho(t, theta, k)
  base = 4*(k - 1);
  [alpha; psi; gamma; omega] = theta((base+1):(base+4));
  y = alpha*sin(2*pi*sigmoid((t - psi)./730, gamma)) + omega;
end
