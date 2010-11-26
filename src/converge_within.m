%%
%% Computes convergence statistic across multiple chains.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

NAME = argv(){1};
a = str2num(argv(){2}); % ensemble number
C = str2num(argv(){3}); % number of chains
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
id = sprintf('%s-%d-converge-within-%d', NAME, C, a);

seq = [ 1:S ]';
halfseq = ceil(seq / 2);

% means and covariances, indexed by chain then iteration
mu = zeros(C, S, N);
Sigma = zeros(C, S, N, N);

for c = 1:C
    filename = sprintf('results/%s-%d-%d.%d.nc.%d', NAME, C, a, ...
                       a, c - 1);
    if exist(filename, "file")
        % read samples
        nc = netcdf(filename, 'r');
        theta = zeros(S, N);
        for n = 1:N
            theta(:,n) = nc{PARAMS{n}}(1:S);
        end

        cum_mu = zeros(S, N);
        cum_Sigma = zeros(S, N, N);

        cum_theta = cumsum(theta, 1);
        cum_mu = (cum_theta(seq,:) - cum_theta(halfseq,:)) ./ repmat(seq - halfseq, 1, N);
        for s = 1:S
            cum_Sigma(s,:,:) = (theta(s,:)'*theta(s,:));
        end
        cum_Sigma = cumsum(cum_Sigma, 1);
        cum_Sigma = cum_Sigma(seq,:,:) - cum_Sigma(halfseq,:,:);
        for s = 1:S
            cum_Sigma(s,:,:) /= seq(s) - halfseq(s) - 1;
            cum_Sigma(s,:,:) -= shiftdim(cum_mu(s,:)'*cum_mu(s,:), -1);
        end
            
        % store in big structure for later
        mu(c,:,:) = cum_mu;
        Sigma(c,:,:,:) = cum_Sigma;
    else
        warning(sprintf('No file %s', filename));
    end
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
            R(s) = (s - 1)/s + (C + 1)/C*lambda1;
        end
    else
        R(s) = 0.0;
    end
end

% output
dlmwrite(sprintf('results/%s.csv', id), real(R), '\t'); 
