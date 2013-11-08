function d = dsigmoid_da(x, a)
  k = 2.*a .- 2;
  x1 = (1 .- x).*x;
  d = 2.^(k + 1).*x1.^k.*log(x1) .+ log(2).*2.^(k + 1).*x1.^k;
end

