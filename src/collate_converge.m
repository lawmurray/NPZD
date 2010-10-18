%%
%% Computes mean and covariance of acceptance rate and convergence
%% statistic across multiple runs.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

DIR = '/home/mur387/sandbox_dmcmc/npzd';
NAMES = { 'dmcmc-share'; 'dmcmc-noshare'; 'haario'; 'straight' };
NODES = { 2, 4, 8, 16 };

for i = 1:length(NAMES)
    name = NAMES{i};
    for j = 1:length(NODES)
        nodes = NODES{j};

        c = 1;
        for k = 0:19
            filename = sprintf('%s/results/%s-%d-converge-%d.csv', DIR, name, ...
                nodes, k);
            if exist(filename, "file")
                tmp = real(dlmread(filename));
                if length(tmp) == 25000 && tmp(end) < 10
                    converge(:,c) = tmp;
                    ++c;
                else
                    warning('Broken file: %s, length = %d, last = %f', ...
                            filename, length(tmp), tmp(end));
                end
            else
                warning('No file: %s', filename);
            end
        end

        mu = real(mean(converge, 2));
        sigma = real(std(converge, 0, 2));
        filename = sprintf('%s/results/%s-%d-converge.csv', DIR, name, ...
                nodes);
        
        is = find(isfinite(mu) .* isfinite(sigma));
        first = 25001 - length(is);
        
        dlmwrite(filename, [ [first:25000]', mu(is), sigma(is) ], '\t');
    end
end
