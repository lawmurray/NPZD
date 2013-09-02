function [nlZ dnlZ] = gpwrap(hyp, inf, mean, cov, lik, x, y, xs, ys)
  [nlZ dnlZ] = gp(hyp, inf, mean, cov, lik, x, y);
  dnlZ.lik(end) = 0.0; % likelihood variance fixed
end
