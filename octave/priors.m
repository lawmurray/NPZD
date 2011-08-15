function [mu0, sd0] = priors ()
    mu0 = [
    % parameters
    log(0.03);
    log(0.02);
    5.0;
    log(0.5);
    log(0.2);
    log(0.2);
    log(1.2);
    log(0.03);
    log(0.25);
    log(0.3);
    log(4.7);
    log(0.2);
    log(0.32);
    log(0.1);
    
    % d-nodes
    log(0.01);
    log(1.2);
    log(0.033);
    log(0.25);
    log(0.12);
    log(4.7);
    log(0.2);
    log(0.32);
    log(0.1);
    log(0.01);
    log(1.1);
    log(0.3531);
    log(0.3531);
   
    % c-nodes
    log(1.7763);
    log(3.7753);
    log(2.9182);
    log(187.8515);
    ];
    
    sd0 = [
    % parameters
    0.2;
    0.3;
    1.0;
    0.1;
    0.8;
    0.8;
    0.63;
    0.37;
    0.3;
    1.0;
    0.7;
    1.3;
    0.25;
    0.5;
    1.0;
    
    % d-nodes
    0.1;
    0.1;
    0.1;
    0.1;
    0.1;
    0.1;
    0.1;
    0.1;
    0.1;
    1.1;
    0.3;
    
    % c-nodes
    0.3;
    0.3;
    0.3;
    0.3;
    ];
end
