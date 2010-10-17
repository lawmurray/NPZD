%%
%% Computes mean and covariance of acceptance rate and convergence
%% statistic across multiple runs.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

DIR = '/home/lawrence/work/workspace/npzd';
NAMES = { 'dmcmc-share'; 'dmcmc-noshare'; 'haario'; 'straight' };
NODES = { 2, 4, 8, 16 };

for i = 1:length(NAMES)
    name = NAMES{i};
    for j = 1:length(NODES)
        nodes = NODES{j};

        for k = 0:9
            filename = sprintf('%s/results/%s-%d-converge-%d.csv', DIR, name, ...
                nodes, k);
            accept(:,k + 1) = dlmread(filename);
        end

        mu = mean(accept, 2);
        sigma = std(accept, 0, 2);
        filename = sprintf('%s/results/%s-%d-converge.csv', DIR, name, ...
                nodes);
        
        is = find(isfinite(mu) .* isfinite(sigma));
        first = 25001 - length(is);
        
        dlmwrite(filename, [ [first:25000]', mu(is), sigma(is) ], '\t');
    end
end
