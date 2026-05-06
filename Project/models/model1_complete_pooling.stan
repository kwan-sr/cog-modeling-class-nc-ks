functions {
  // Shifted-Wald (inverse Gaussian) log-PDF for single-boundary
  // first-passage time of a diffusion process with drift v > 0
  // crossing threshold a > 0, shifted by non-decision time tau.
  real shifted_wald_lpdf(real y, real v, real a, real tau) {
    real t = y - tau;
    if (t <= 0) return negative_infinity();
    return log(a) - 0.5 * log(2 * pi()) - 1.5 * log(t) - square(a - v * t) / (2 * t);
  }

  // Inverse Gaussian RNG (Michael, Schucany & Haas, 1976)
  real inv_gaussian_rng(real mu, real lambda) {
    real nu = normal_rng(0, 1);
    real y = square(nu);
    real x = mu + (square(mu) * y) / (2 * lambda)
             - (mu / (2 * lambda))
               * sqrt(4 * mu * lambda * y + square(mu) * square(y));
    real z = uniform_rng(0, 1);
    if (z <= mu / (mu + x))
      return x;
    else
      return square(mu) / x;
  }
}

data {
  int<lower=1> N;                          // total number of trials
  array[N] int<lower=1, upper=4> cond;     // 1=Dfast, 2=Dslow, 3=NDfast, 4=NDslow
  vector<lower=0>[N] y;                    // RT in seconds
  real<lower=0> min_y;                     // minimum RT (for tau upper bound)
}

parameters {
  vector<lower=0>[4] v;                    // drift rates per condition
  vector<lower=0>[4] a;                    // thresholds per condition
  real<lower=0, upper=min_y> tau;          // non-writing time
}

model {
  v ~ gamma(3, 1);            // mean = 3
  a ~ gamma(3, 1);            // mean = 3
  tau ~ gamma(2, 20);         // mean = 0.1s

  // Likelihood
  for (n in 1:N) {
    y[n] ~ shifted_wald(v[cond[n]], a[cond[n]], tau);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = shifted_wald_lpdf(y[n] | v[cond[n]], a[cond[n]], tau);
    y_rep[n] = tau + inv_gaussian_rng(a[cond[n]] / v[cond[n]],
                                       square(a[cond[n]]));
  }
}