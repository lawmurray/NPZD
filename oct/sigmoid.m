function y = sigmoid(x, a)
    y = (2.*x.*(1 .- x)).^(2.*a .- 2)./(0.5.^(2.*a .- 2));
end
