function plot_state()
    subplot(5,1,1);
    bi_plot_quantiles('results/posterior.nc', 'N');
    hold on;
    ylabel('N');
    bi_plot_paths('data/obs.nc', 'N_obs');
    hold off;
    ax = axis();
    axis([0 4*365 ax(3) ax(4)]);
    grid on;

    subplot(5,1,2);
    bi_plot_quantiles('results/posterior.nc', 'P');
    ylabel('P');
    ax = axis();
    axis([0 4*365 ax(3) ax(4)]);
    grid on;

    subplot(5,1,3);
    bi_plot_quantiles('results/posterior.nc', 'Z');
    ylabel('Z');
    ax = axis();
    axis([0 4*365 ax(3) ax(4)]);
    grid on;

    subplot(5,1,4);
    bi_plot_quantiles('results/posterior.nc', 'D');    
    ylabel('D');
    ax = axis();
    axis([0 4*365 ax(3) ax(4)]);
    grid on;

    subplot(5,1,5);
    bi_plot_quantiles('results/posterior.nc', 'Chla');
    hold on;
    xlabel('t');
    ylabel('Chla');
    bi_plot_paths('data/obs.nc', 'Chla_obs');
    hold off;
    ax = axis();
    axis([0 4*365 ax(3) ax(4)]);
    grid on;
end
