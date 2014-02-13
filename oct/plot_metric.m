function plot_metric(k)
    if nargin < 1
        k = 1;
    end
    
    filters = {'Bootstrap'; 'Bridge'};
    ylabels = {
        'ESS/N';
        'CAR';
        'Mean -log p(x_n|x_0)';
        'Std. dev. log p(x_n|x_0)'
        'Mean execution time (s)';
        'Std. dev. execution time (s)';
        'Number of samples';
        };
        
    logPs = [5:10]';
    Ps = 2.**logPs;

    y = zeros(length(Ps), length(filters));
    ly = zeros(size(y));
    uy = zeros(size(y));
    
    for i = 1:length(filters)
        for j = 1:length(Ps)
            P = Ps(j);
            file = sprintf('%s-%d.csv', tolower(filters{i}), P);
            %X = dlmread(file);
            % ^ dlmread doesn't handle the headers with spaces
            X = textread(file, '%f', inf, 'headerlines', 1);
            X = reshape(X, 7, length(X)/7)';
            
            if k == 3
                X = -X; % convert log-likelihood to -ve log-likelihood
            end
            if k == 5 || k == 6
                X = X/1e6; % convert to seconds
            end
            
            y(j,i) = quantile(X(:,k), 0.5);
            ly(j,i) = quantile(X(:,k), 0.25);
            uy(j,i) = quantile(X(:,k), 0.75);
        end
    end

    % plot bars
    h = bar (logPs, y);
    for i = 1:length(h)
        set (h(i), 'facecolor', fade(watercolour(i), 0.5));
        set (h(i), 'edgecolor', watercolour(i));
    end

    % plot errors
    % errorbar seems to produce raster lines on SVG output, do manually
    hold on;
    for i = 1:columns(y)
        for j = 1:rows(y)
            if i == 1
                x = logPs(j) - 0.2;
            else
                x = logPs(j) + 0.2;
            end
            line ([x x], [ly(j,i), uy(j,i)], 'color', watercolour(i), ...
                'linewidth', 2);
            line ([x - 0.1, x + 0.1], [ly(j,i) ly(j,i)], 'color', ...
                watercolour(i), 'linewidth', 2);
            line ([x - 0.1, x + 0.1], [uy(j,i) uy(j,i)], 'color', ...
                watercolour(i), 'linewidth', 2);
        end
    end
    
    grid on;
    legend(filters, 'location', 'northwest');
    ylabel(ylabels{k});
    xlabel('log_2 N');
    ax = axis();
    axis([logPs(1) - 0.5, logPs(end) + 0.5, ax(3), ax(4)]);
    hold off;
end
