function d = dsigmoid_dx(x, a)
  k = 2.*a .- 2;
  d = (k.*2.^k.*(1 .- 2.*x).*((1 .- x).*x).^k)./((1 .- x).*x);
end

