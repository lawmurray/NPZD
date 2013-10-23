function y = ess(LL)
  mx = max(LL, [], 2);
  L = exp(LL - repmat(mx, 1, columns(LL)));
  y = sum(L, 2).^2./sum(L.^2, 2);
end
