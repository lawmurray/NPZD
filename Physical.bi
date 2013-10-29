/**
 * Physical variable input into the biological model.
 * 
 * @author Emlyn Jones <emlyn.jones@csiro.au>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
model Physical {
  dim n(3);
  dim z(size = 700, boundary = 'repeat');

  /* mean function parameters */
  param alpha[n];
  param psi[n];
  param beta[n];
  param a;
  param c;

  /* autoregressive parameters */
  param gamma;
  param sigma_x;

  /* observation parameters */
  const sigma_y = 0.5;

  /* mean function */
  state phi[n];
  state mu[z];

  /* state */
  state t
  state x[z];
  noise dW[z];

  /* observations */
  obs T[z];

  /* constraints */
  const min_T = 0.0;
  const max_T = 40.0;

  sub parameter {
    alpha[i] ~ uniform(min_T, max_T);
    psi[i] ~ uniform(0.0, 365.0);
    beta[i] ~ uniform(min_T, max_T);
    a ~ gaussian(0.0, 1.0);
    c ~ uniform(min_T, max_T);
    gamma ~ uniform(0.5, 1.0);
    sigma_x ~ inverse_gamma(2.0, 3.0);
  }

  sub proposal_parameter {
    alpha[n] ~ truncated_gaussian(alpha[n], 0.05, min_T, max_T);
    psi[n] ~ truncated_gaussian(psi[n], 0.5, 0.0, 365.0);
    beta[n] ~ truncated_gaussian(beta[n], 0.05, min_T, max_T);
    a ~ gaussian(a, 0.01);
    c ~ truncated_gaussian(c, 0.1, min_T, max_T);
    gamma ~ truncated_gaussian(gamma, 0.01, 0.5, 1.0);
    sigma_x ~ inverse_gamma(8.0, 9.0*sigma_x);
  }

  sub initial {
    t <- 0
    x[z] ~ gaussian(10.0, 5.0);
  }

  sub proposal_initial {
    t <- 0
    x[z] ~ truncated_gaussian(x[z], 0.1, min_T, max_T);
  }

  sub transition(delta = 0.1) {
    /* mean function */
    phi[n] <- alpha[n]*sin((t - psi[n])/365.0) + beta[n];
    mu[z] <- phi[1]*0.5*(sinh((phi[2] - z)/phi[3])/cosh((phi[2] - z)/phi[3]) + 0.5) + a*z + c;

    /* residuals */
    dW[z] ~ gaussian(0.0, sqrt(0.1));
    ode(alg = 'RK4(3)', h = 0.1) {
      dx[z]/dt = gamma*x[z] + 0.5*(1.0 - gamma)*(x[z-1] + x[z+1]) + sigma_x*dW[z];
    }

    /* time */
    t <- t + 0.1
  }

  sub observation {
    T[z] ~ gaussian(mu[z] + x[z], sigma_y);
  }
}
