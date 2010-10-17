%%
%% Computes mean and covariance of acceptance rate and convergence
%% statistic across multiple runs.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

DIR = '/home/lawrence/work/workspace/npzd';
NAMES = { 'dmcmc-share'; 'dmcmc-noshare' };
NODES = { 2, 4, 8, 16 };

for i = 1:length(NAMES)
    name = NAMES{i};
    for j = 1:length(NODES)
        nodes = NODES{j};

        for k = 0:9
            filename = sprintf('%s/results/%s-%d-accept-%d.csv', DIR, name, ...
                nodes, k);
            accept(k + 1,:,:) = dlmread(filename);
        end

        mu = mean(accept, 1);
        sigma = std(accept, 1);
        filename = sprintf('%s/results/%s-%d-accept.csv', DIR, name, ...
                nodes);
        dlmwrite(filename, [ mu, sigma ]);
    end
end
