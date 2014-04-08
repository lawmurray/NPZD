function plot_metric(metric)
    pkg load netcdf;
    if nargin < 1
        metric = 1;
    end
    
    y = {};
    filters = {'Bootstrap'; 'Bridge'};
    for i = 1:length(filters)
        y{i} = [];
        rep = 0;
        file = sprintf('results/test_%s-%d.nc', tolower(filters{i}), rep);
        while exist(file, 'file')
            L = ncread(file, 'loglikelihood');
            T = ncread(file, 'time');
            P = ncread(file, 'P');
        
            if metric == 1
                % read 'true' likelihood
                truth_file = sprintf('results/test_exact-%d.nc', rep);
                ll = ncread(truth_file, 'loglikelihood');
                
                tmp = 1.0./mean((L - ll).^2)./mean(T);
            elseif metric == 2
                tmp = ess(L)./mean(T);
            elseif metric == 3
                tmp = car(L)./mean(T);
            end
            y{i} = [ y{i}; tmp ];
            
            rep = rep + 1;
            file = sprintf('results/test_%s-%d.nc', tolower(filters{i}), rep);
        end
    end

    for i = 1:rows(y{1})
        for j = 1:columns(y{1})
            if y{1}(i,j) >= 2*y{2}(i,j)
                col = watercolour(1);
            elseif y{2}(i,j) >= 2*y{1}(i,j)
                col = watercolour(2);
            else
                col = [0.7 0.7 0.7];
            end
            h = loglog(y{1}(i,j), y{2}(i,j), '.', 'color', col, ...
                'markersize', sqrt(3*P(j)/pi));
            hold on;
        end
    end

    %axis('tight');
    ax = axis();
    mn = min([ax(1), ax(3)]);
    mx = max([ax(2), ax(4)]);
    loglog([mn mx], [mn mx], '--k');
    axis([mn mx mn mx]);
    axis('square');
    grid on;
    grid minor off;
end
