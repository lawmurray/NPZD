function y = sigmoid(x, a)
    k = 2.*a .- 2;
    
    y = (4.*x.*(1 .- x)).^k;
    y(find(x <= 0)) = 0.0;
    y(find(x >= 1)) = 1.0;
end
