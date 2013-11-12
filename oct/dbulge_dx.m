function d = dbulge_dx(x, a)
  k = 2.*a .- 2;
  x1 = (1 .- x).*x;

  d = (k.*4.^k.*(1 .- 2.*x).*x1.^k)./x1;
  d(find(x1 <= 0)) = 0; % limit
end
