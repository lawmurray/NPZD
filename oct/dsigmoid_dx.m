function d = dsigmoid_dx(x, a)
  k = 2.*a .- 2;
  x1 = (1 .- x).*x;
  
  d = (k.*0.5.^k.*2.^k.*(1 .- 2.*x).*x1.^k)./x1;
end
