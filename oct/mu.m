function x = mu(s, t, theta)
  rho5 = rho(t, theta, 5);
  rho6 = rho(t, theta, 6);
  rho7 = rho(t, theta, 7);

  is1 = find(s <= rho5);
  is2 = find((s > rho5).*(s <= rho6));
  is3 = find((s > rho6).*(s <= rho7));
  is4 = find(s > rho7);

  d(is1,:) = mu1(s(is1), t(is1), theta);
  d(is2,:) = mu2(s(is2), t(is2), theta);
  d(is3,:) = mu3(s(is3), t(is3), theta);
  d(is4,:) = mu4(s(is4), t(is4), theta);
end

