function d = dsigmoid_da(x, a)
  k = 2.*a .- 2;
  x1 = 4.*x.*(1 .- x);
  logx1 = log(x1);
  logx1(find(x1 <= 0)) = 0;
  d = 2.*logx1.*x1.^k;
end
