function x = tau(s, t, theta, k)
  mx = rho(t, theta, 5 + k);
  mn = rho(t, theta, 4 + k);

  x = (s .- mn)./(mx .- mn);
  x(find(mx .- mn == 0)) = 0;
end
