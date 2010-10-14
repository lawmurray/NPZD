%%
%% Computes acceptance rate of remote proposals.
%%
%% @author Lawrence Murray <lawrence.murray@csiro.au>
%% $Rev$
%% $Date$
%%

DIR = '/home/mur387/sandbox_dmcmc/npzd';
NAMES = { 'dmcmc-share'; 'dmcmc-noshare' };
NODES = { 4, 8, 16, 32 };

for i = 1:length(NAMES)
    name = NAMES{i};
    for j = 1:length(NODES)
        nodes = NODES{j};
        id = sprintf('%s-%d-accept', name, nodes);

        % build data set
        remote = [];
        logaccept = [];
        for proc = 0:(nodes - 1)
            filename = sprintf('%s/results/%s-%d.nc.%d', DIR, name, nodes, ...
                               proc);
            if exist(filename, 'file')
                nc = netcdf(filename, 'r');
                remote = [ remote, nc{'remote'}(:) ];
                logaccept = [ logaccept, nc{'logaccept'}(:) ];
            else
                warning('File %s does not exist', filename);
            end
        end
    
        % compute acceptance rates
        remote_rowsum = sum(remote, 2);
        logaccept_rowsum = sum(min(remote .* logaccept, 0), 2);
        remote_cumsum = cumsum(remote_rowsum);
        logaccept_cumsum = cumsum(logaccept_rowsum);
        N = length(remote_cumsum);
    
        seq = [ 1:N ];
        halfseq = ceil(seq / 2);
        y = (logaccept_cumsum(seq) - logaccept_cumsum(halfseq)) ./ ...
            (remote_cumsum(seq) - remote_cumsum(halfseq));
        
        % output
        dlmwrite(sprintf('%s/results/%s.csv', DIR, id), y(200:end), '\t');
    end
end
