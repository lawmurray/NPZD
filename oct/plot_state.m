function plot_state()
    ps = [25000:100000];

    subplot(2,2,1);
    bi_plot_quantiles('results/prior.nc', 'N');
    hold on;
    bi_plot_quantiles('results/posterior.nc', 'N', [], ps);
    ylabel('N');
    bi_plot_paths('data/obs.nc', 'N_obs');
    hold off;
    
    subplot(2,2,3);
    bi_plot_quantiles('results/prior.nc', 'Chla');
    hold on;
    bi_plot_quantiles('results/posterior.nc', 'Chla', [], ps);
    xlabel('t');
    ylabel('Chla');
    bi_plot_paths('data/obs.nc', 'Chla_obs');
    hold off;
    
    subplot(3,2,2);
    bi_plot_quantiles('results/prior.nc', 'P');
    hold on;
    bi_plot_quantiles('results/posterior.nc', 'P', [], ps);
    ylabel('P');
    
    subplot(3,2,4);
    bi_plot_quantiles('results/prior.nc', 'Z');
    hold on;
    bi_plot_quantiles('results/posterior.nc', 'Z', [], ps);
    ylabel('Z');
    
    subplot(3,2,6);
    bi_plot_quantiles('results/prior.nc', 'D');
    hold on;
    bi_plot_quantiles('results/posterior.nc', 'D', [], ps);    
    ylabel('D');
    xlabel('t');
end
