function d = dmu4_dtheta(s, t, theta)
  d = zeros(length(s), length(theta));
  d(:,end) = 1;
end
