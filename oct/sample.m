% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} sample (model, n)
%
% Sample forcings from Gaussian process model.
% @end deftypefn
%
function m = sample(m, n)
    ts = [m.t; m.u; m.v];
    nts = length(ts);
    
    nt = length(m.t);
    nu = length(m.u);
    nv = length(m.v);

    m.ux = zeros(n, nu);
    m.vx = zeros(n, nv);

    mu = feval(m.meanfunc{:}, m.hyp.mean, ts); % mean
    K = feval(m.covfunc{:}, m.hyp.cov, ts); % covariance
    x = zeros(nts, 1);
        
    % predict trajectories
    for i = 1:n
        K(1:nt,1:nt) += diag(repmat(exp(2*m.hyp.lik), nt, 1), nt, nt); % nugget on obs
        invK = cholinv(K(1:nt,1:nt)); % first inversion, rest recursive
        x(1:nt) = m.y - mu(1:nt);

        for j = 1:(nu + nv)
            l = nt + j - 1;
            k = K(1:l,l + 1);
            w = invK*k;
            mu1 = x(1:l)'*w;
            sigma1 = sqrt(k'*w);
            x(l + 1) = normrnd(mu1, sigma1);

            % recursive update of invK
            a = -invK*k;
            b = k'*a;
            invK = [invK + a*(b*a'), a*b; b*a', b];
        end
        x += mu;
        
        m.ux(i,:) = x((nt + 1):(nt + nu));
        m.vx(i,:) = x((nt + nu + 1):end);
    end

end
