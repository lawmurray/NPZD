function y = sigmoid(x, a)
    k = 2.*a .- 2;
    y = (4.*x.*(1 .- x)).^k;
end
