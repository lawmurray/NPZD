function d = dsigmoid_dx(x, a)
  d = betapdf(x, a, a);
end
