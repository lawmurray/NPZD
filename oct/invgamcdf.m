function c = invgamcdf(x, a, b)
    c = gammainc(a, b./x)/gamma(a);
end
