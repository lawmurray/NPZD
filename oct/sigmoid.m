function y = sigmoid(x, a)
  y = betacdf(x, a, a);
end
