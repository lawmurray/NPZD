function y = squash(x, n)
  y = zeros(size(x));
  x = mod(x, 2*pi);
  
  is1 = find((x >= 0).*(x <= pi/2));
  is2 = find((x > pi/2).*(x <= pi));
  is3 = find((x > pi).*(x <= 3*pi/2));
  is4 = find((x > 3*pi/2).*(x < 2*pi));
  
  y(is1) = (1 - 1./(tan(x(is1)).^n + 1)).^(1/n);
  y(is2) = (1 - 1./(tan(pi - x(is2)).^n + 1)).^(1/n);
  y(is3) = -1*(1 - 1./(tan(x(is3) - pi).^n + 1)).^(1/n);
  y(is4) = -1*(1 - 1./(tan(2*pi - x(is4)).^n + 1)).^(1/n);
end
