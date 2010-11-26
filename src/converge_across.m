%%
%% Computes convergence statistic across multiple ensembles of chains.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

NAME = argv(){1};
A = str2num(argv(){2}); % number of ensembles
C = str2num(argv(){3}); % number of chains in each ensemble
S = str2num(argv(){4}); % number of steps
PARAMS = {
    "Kw";
    "KCh";
    "Dsi";
    "ZgD";
    "PDF";
    "ZDF";
    "muPgC";
    "muPgR";
    "muPCh";
    "muPaN";
    "muPRN";
    "muZin";
    "muZCl";
    "muZgE";
    "muDre";
    "muZmQ"
};

N = length(PARAMS);

% means and covariances, indexed by arrayid then iteration
mu = zeros(A, S, N);
Sigma = zeros(A, S, N, N);

seq = [ 1:S ]';
halfseq = ceil(seq / 2);

id = sprintf('%s-%d-converge-across', NAME, C);
for a = 1:A
    % read samples
    theta = zeros(C, S, N);
    for c = 1:C
        filename = sprintf('results/%s-%d-%d.%d.nc.%d', NAME, ...
                           C, a - 1, a - 1, c - 1);
        if exist(filename, "file")
            nc = netcdf(filename, 'r');
            for n = 1:N
                theta(c,:,n) = nc{PARAMS{n}}(1:S);
            end
        else
            error('No file %s', filename);
        end
    end
    
    % compute ensemble mean and covariance for this run at each step
    cum_theta = squeeze(cumsum(sum(theta, 1), 2));
    cum_mu = (cum_theta(seq,:) - cum_theta(halfseq,:)) ./ repmat(C*(seq - halfseq), 1, N);
    cum_Sigma = zeros(S, N, N);
    for c = 1:C
        for s = 1:S
            x = squeeze(theta(c,s,:));
            cum_Sigma(s,:,:) += shiftdim(x*x', -1);
        end
    end
    
    cum_Sigma = cumsum(cum_Sigma, 1);
    cum_Sigma = cum_Sigma(seq,:,:) - cum_Sigma(halfseq,:,:);
    for s = 1:S
        cum_Sigma(s,:,:) /= C*(seq(s) - halfseq(s)) - 1;
        cum_Sigma(s,:,:) -= shiftdim(cum_mu(s,:)'*cum_mu(s,:), -1);
    end
    
    % store in big structure for later
    mu(a,:,:) = cum_mu;
    Sigma(a,:,:,:) = cum_Sigma;
end

% compute W and B/n
W = squeeze(mean(Sigma, 1));
Bon = zeros(S, N, N);
for s = 1:S
    Bon(s,:,:) = cov(squeeze(mu(:,s,:)));
end

% scalar comparison
R = zeros(S,1);
for s = 1:S
    [ U, posdef ] = chol(squeeze(W(s,:,:)));
    if posdef == 0
        Wsi = chol2inv(U);
        Bonp = squeeze(Bon(s,:,:));
        
        % eig causes error with any non-finite values
        if sum(isfinite(Wsi(:))) == N*N && ...
                sum(isfinite(Bonp(:))) == N*N
            lambda1 = max(eig(Wsi*Bonp));
            R(s) = (C*s - 1)/(C*s) + (A + 1)/A*lambda1;
        end
    else
        R(s) = 0.0;
    end
end

% output
dlmwrite(sprintf('results/%s.csv', id), real(R), '\t'); 
