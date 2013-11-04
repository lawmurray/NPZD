/**
 * Physical variable input into the biological model.
 * 
 * @author Emlyn Jones <emlyn.jones@csiro.au>
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 */
model PhysicalRaster {
  const h = 1.0/24.0;

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
  state mu;

  /* state */
  state r; // rasterised index
  state t; // time index
  state s; // spatial index
  state x[z];
  state x_1[z](has_output = 0);
  state x_new(has_output = 0);
  noise eps_new(has_output = 0);

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
    alpha[0] ~ truncated_gaussian(alpha[0], 0.05, min_T, max_T);
    alpha[1] ~ truncated_gaussian(alpha[1], 0.05, min_T, max_T);
    alpha[2] ~ truncated_gaussian(alpha[2], 0.05, min_T, max_T);
    psi[n] ~ truncated_gaussian(psi[n], 0.5, 0.0, 365.0);
    beta[n] ~ truncated_gaussian(beta[n], 0.05, min_T, max_T);
    a ~ gaussian(a, 0.01);
    c ~ truncated_gaussian(c, 0.1, min_T, max_T);
    gamma ~ truncated_gaussian(gamma, 0.01, 0.5, 1.0);
    sigma_x ~ inverse_gamma(8.0, 9.0*sigma_x);
  }

  sub initial {
    r <- 0;
    t <- 0;
    s <- 0;
    x[z] ~ gaussian(10.0, 5.0);
  }

  sub proposal_initial {
    r <- 0;
    t <- 0;
    s <- 0;
    x[z] ~ truncated_gaussian(x[z], 0.1, min_T, max_T);
  }

  sub transition(delta = h) {
    t <- h*floor(r/700.0);
    s <- mod(r, 700.0);
    r <- r + 1;
    x_1 <- (s == 0) ? x : x_1;

    /* residuals */
    eps_new ~ gaussian(0.0, sigma_x*sqrt(h));
    x_new <- x_1[s] - gamma*x_1[s] + 0.5*gamma*(x_1[s-1] + x_1[s+1]) + eps_new;
    x[i] <- (i == s) ? x_new : x[i];
  }

  sub observation {
    /* mean function */
    phi[n] <- alpha[n]*sin((t - psi[n])/365.0) + beta[n];
    mu <- phi[1]*0.5*(sinh((phi[2] - s)/phi[3])/cosh((phi[2] - s)/phi[3]) + 0.5) + a*s + c;

    T ~ gaussian(mu + x_new, sigma_y);
  }
}
