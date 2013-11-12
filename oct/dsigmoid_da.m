function d = dsigmoid_da(x, a)
  % finite difference estimate, otherwise need digamma and hypergeometric funcs
  eps = 1.0e-8;
  d = (sigmoid(x, a .+ eps) .- sigmoid(x, a .- eps))/(2.*eps);
end
