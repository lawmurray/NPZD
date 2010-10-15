%%
%% Computes convergence of multiple chains.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

DIR = '/home/mur387/sandbox_dmcmc/npzd';
NAMES = { 'dmcmc-share', 'dmcmc-noshare', 'haario', 'straight' };
NODES = { 4, 8, 16 };
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

for i = 1:length(NAMES)
    name = NAMES{i};

    % means and covariances, indexed by chain then iteration
    mu = [];
    Sigma = [];

    for j = 1:length(NODES)
        C = NODES{j};
        id = sprintf('%s-%d-converge', name, C);

        for proc = 0:(C - 1)
            filename = sprintf('%s/results/%s-%d.nc.%d', DIR, name, C, proc);
            
            nc = netcdf(filename, 'r');
            theta = [];
            for k = 1:N
                param = PARAMS{k};
                theta = [ theta, nc{param}(:) ];
            end
            
            P = rows(theta);
            seq = [ 1:P ]';
            halfseq = ceil(seq / 2);
            
            cum_mu = cumsum(theta, 1) ./ repmat(seq, 1, N);
            cum_Sigma = zeros(P, N, N);
            for k = 2:P % omit first, can't compute covariance
                cum_Sigma(k,:,:) = (theta(k,:)'*theta(k,:)) ./ (k - 1);
            end
            cum_Sigma = cumsum(cum_Sigma, 1);
            for k = 2:P
                cum_Sigma(k,:,:) -= shiftdim(cum_mu(k,:)'*cum_mu(k,:), -1);
            end
            
            % store in big structure for later
            mu(proc + 1,:,:) = cum_mu;
            Sigma(proc + 1,:,:,:) = cum_Sigma;
        end
        
        % compute W and B/n
        W = squeeze(mean(Sigma, 1));
        Bon = zeros(P, N, N);
        for p = 1:P
            Bon(p,:,:) = cov(squeeze(mu(:,p,:)));
        end
    
        % scalar comparison
        for p = 1:P
            Wpi = inv(squeeze(W(p,:,:)));
            Bonp = squeeze(Bon(p));
        
            % eig causes error with any non-finite values
            if cumsum(cumsum(isfinite(Wpi), 1), 2) > 0 && ...
                    cumsum(cumsum(isfinite(Bonp), 1), 2) > 0
                lambda1 = max(eig(Wpi*Bonp));
                Rp(p) = (P - 1)/P + (C + 1)/C*lambda1;
            else
                Rp(p) = 0;
            end
        end
       
        % output
        dlmwrite(sprintf('%s/results/%s.csv', DIR, id), Rp', '\t'); 
    end
end
