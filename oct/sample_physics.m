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
function m = sample_physics(m, n)
    ts = [m.t; m.u];
    nts = length(ts);
    
    nt = length(m.t);
    nu = length(m.u);

    mu = feval(m.meanfunc{:}, m.hyp.mean, ts); % mean
    K = feval(m.covfunc{:}, m.hyp.cov, ts); % covariance
    X = zeros(nts, n);
        
    % predict trajectories
    K(1:nt,1:nt) += diag(repmat(exp(2*m.hyp.lik), nt, 1), nt, nt); % nugget on obs
    invK = cholinv(K(1:nt,1:nt)); % first inversion, rest recursive
    X(1:nt,:) = repmat(m.y - mu(1:nt), 1, n);

    for j = 1:nu
        l = nt + j - 1;
        k = K(1:l,l + 1);
        w = invK*k;
        mu1 = X(1:l,:)'*w;
        s1 = sqrt(K(l + 1, l + 1) - k'*w);
        if iscomplex(s1)
            X(l + 1,:) = mu1;
        else
            X(l + 1,:) = normrnd(mu1, s1);
        end

        % recursive update of invK
        a = -invK*k;
        b = 1.0/(K(l + 1, l + 1) + k'*a);
        invK = [invK + a*(b*a'), a*b; b*a', b];
    end
    X += repmat(mu, 1, n);
    m.X = X((nt + 1):(nt + nu),:)';
end
