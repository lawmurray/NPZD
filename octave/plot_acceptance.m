% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} prepare_acceptance ()
%
% Plot acceptance rate comparision for NPZD model.
% @end deftypefn
%
function plot_acceptance ()
    experiments = { 1, 2, 6 };
    load model_acceptance

    ax1 = zeros(1,4);
    cax1 = zeros(1,2);
    ax2 = zeros(1,4);
    cax2 = zeros(1,2);
    for i = 1:length (experiments)
        j = experiments{i};
        
        subplot(2,3,i);
        polar_likelihood(models{j}, [ mxs{j}(2:end,:); mns{j} ], ...
            mxs{j}(1,:), 10);
        ax1 = max([ ax1 abs(axis()) ]);
        cax1 = min([ cax1, caxis() ]);
        
        subplot(2,3,i + 3);
        polar_likelihood(models{j}, [ mxs{j}; mns{j}(2:end,:) ], ...
            mns{j}(1,:), 10);
        ax2 = max([ ax2 abs(axis()) ]);
        cax2 = min([ cax2, caxis() ]);
    end
    
    for i = 1:3
        subplot(2,3,i);
        axis([-ax1 ax1 -ax1 ax1], 'square');
        caxis([ cax1 0 ]);
    
        subplot(2,3,i + 3);
        axis([-ax2 ax2 -ax2 ax2], 'square');
        caxis([ cax2 0 ]);
    end
end
