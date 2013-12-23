/**
 * Physical variable input into the biological model.
 * 
 * @author Emlyn Jones <emlyn.jones@csiro.au>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
model Physical {
  dim n(3);
  dim z(size = 700, boundary = 'extended');

  const h = 1.0/24.0;
  const pi = 3.141592653589793;

  /* mean function parameters */
  param alpha[n];
  param psi[n];
  param beta[n];
  param a;
  param c;

  /* autoregressive parameters */
  param gamma;
  param sigma2_x;

  /* observation parameters */
  const sigma_y = 0.02;

  /* state */
  state t(has_output = 0);
  state x[z];
  noise eps[z](has_output = 0);

  /* observations */
  obs T[z];

  sub parameter {
    /*alpha[i] ~ gaussian(0.0, 2.0);
    psi[i] ~ uniform(-182.5, 182.5);
    beta[0] ~ truncated_gaussian(0.0, 5.0, 0.0);
    beta[1] ~ gaussian(0.0, 100.0);
    beta[2] ~ uniform(0, 700);
    a ~ gaussian(0.0, 1.0);
    c ~ gaussian(0.0, 5.0);*/
    gamma ~ gamma(2.0, 2.0);
    sigma2_x ~ inverse_gamma(2.0, 0.01);
  }

  sub proposal_parameter {
    /*alpha[0] ~ gaussian(alpha[0], 1.0e-3);
    alpha[1] ~ gaussian(alpha[1], 1.0e-4);
    alpha[2] ~ gaussian(alpha[2], 1.0e-5);
    psi[0] ~ gaussian(psi[0], 1.0e-4);
    psi[1] ~ gaussian(psi[1], 1.0e-7);
    psi[2] ~ gaussian(psi[2], 1.0e-9);
    beta[0] ~ gaussian(beta[0], 1.0e-3);
    beta[1] ~ gaussian(beta[1], 1.0e-3);
    beta[2] ~ gaussian(beta[2], 1.0e-3);

    a ~ gaussian(a, 1.0e-7);
    c ~ gaussian(c, 1.0e-4);*/
    gamma ~ truncated_gaussian(gamma, 1.0e-3, 0.0);
    sigma2_x ~ inverse_gamma(16.0, 17.0*sigma2_x);
  }

  sub initial {
    t <- 0;
    //x[z] ~ gaussian(0.0, 0.01);
    x[z] <- 0
  }

  sub proposal_initial {
    x[z] ~ gaussian(x[z], 0.01);
  }

  sub transition(delta = h) {
    eps[z] ~ gaussian(0.0, sqrt(sigma2_x*h));
    x[z] <- x[z] - h*gamma*(x[z] - 0.5*(x[z-1] + x[z+1])) + eps[z];
    t <- t + h;
  }

  sub observation {
    inline phi0 = alpha[0]*sin(2*pi*(t - psi[0])/365.0) + beta[0];
    inline phi1 = alpha[1]*sin(2*pi*(t - psi[1])/365.0) + beta[1];
    inline phi2 = alpha[2]*sin(2*pi*(t - psi[2])/365.0) + beta[2];

    T[z] ~ log_gaussian(phi0*(0.5*tanh((phi1 - z)/phi2) + 0.5) + a*z + c + x[z], sigma_y);
  }
}
