% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>

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

    % joint distribution
    mu = feval(m.meanfunc{:}, m.hyp.mean, ts);
    K = feval(m.covfunc{:}, m.hyp.cov, ts);
    K(1:nt,1:nt) += diag(repmat(exp(2*m.hyp.lik), nt, 1), nt, nt); % nugget
    
    % posterior distribution
    y = m.y;
    mu1 = mu(1:nt);
    mu2 = mu((nt+1):end);
    K1 = K(1:nt,1:nt);
    K2 = K((nt+1):end,(nt+1):end);
    C = K(1:nt,(nt+1):end);
    U1 = chol(K1);
    %invK1 = cholinv(K1);
    mu2 = mu2 + C'*(U1\(U1'\(y - mu1)));
    A = U1'\C;
    K2 = K2 - A'*A;
        
    [U2,pos] = chol(K2);
    if pos != 0
        % use diagonalisation instead
        [E,Lambda] = eig(K2);
        Lambda = max(Lambda, 0.0);
        U2 = sqrt(Lambda)*E';
        
        % ...this is also the fullback in the statistics package, which
        % could be used like this:
        %pkg load statistics
        %m.X = real(mvnrnd(mu2', K2, n, 0));
    end
    
    z = normrnd(0.0, 1.0, [nu, n]);
    X = real(repmat(mu2, 1, n) + U2'*z);
   
    m.X = X';
end
