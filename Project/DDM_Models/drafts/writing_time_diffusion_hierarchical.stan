functions {
  real shifted_wald_lpdf(real y, real v, real a, real tau) {
    real t = y - tau;
    if (t <= 0) return negative_infinity();
    return log(a) - 0.5 * log(2 * pi()) - 1.5 * log(t)
           - square(a - v * t) / (2 * t);
  }
}

data {
  int<lower=1> N;                           // total trials
  int<lower=1> N_subj;                      // number of participants
  array[N] int<lower=1, upper=4> cond;      // condition: 1=Dfast, 2=Dslow, 3=NDfast, 4=NDslow
  array[N] int<lower=1, upper=N_subj> subj; // participant index per trial
  vector<lower=0>[N] y;                     // RT in seconds
  real<lower=0> min_y;                      // minimum RT (tau upper bound)
}

parameters {
  // Population-level condition means
  vector<lower=0>[4] mu_v;
  vector<lower=0>[4] mu_a;

  // Subject-level random effects (non-centered, on log scale)
  vector[N_subj] z_v_raw;
  vector[N_subj] z_a_raw;
  real<lower=0> sigma_v;       // SD of log-drift across subjects
  real<lower=0> sigma_a;       // SD of log-threshold across subjects

  real<lower=0, upper=min_y> tau;
}

transformed parameters {
  // Scaled subject offsets (log-multiplicative)
  vector[N_subj] z_v = z_v_raw * sigma_v;
  vector[N_subj] z_a = z_a_raw * sigma_a;
}

model {
  // Population-level priors (same as before)
  mu_v ~ gamma(3, 1);
  mu_a ~ gamma(3, 1);

  // Random-effect SDs: half-normal, modest
  sigma_v ~ normal(0, 0.5);
  sigma_a ~ normal(0, 0.5);

  // Standardized subject offsets
  z_v_raw ~ std_normal();
  z_a_raw ~ std_normal();

  // Non-decision time
  tau ~ gamma(2, 20);

  // Likelihood: each trial uses the participant's offset on top of the condition mean
  for (n in 1:N) {
    real v_n = mu_v[cond[n]] * exp(z_v[subj[n]]);
    real a_n = mu_a[cond[n]] * exp(z_a[subj[n]]);
    y[n] ~ shifted_wald(v_n, a_n, tau);
  }
}

// generated quantities {
//   vector[N] log_lik;
//   vector[N] y_rep;
//   for (n in 1:N) {
//     real v_n = mu_v[cond[n]] * exp(z_v[subj[n]]);
//     real a_n = mu_a[cond[n]] * exp(z_a[subj[n]]);
//     log_lik[n] = shifted_wald_lpdf(y[n] | v_n, a_n, tau);
//     y_rep[n] = tau + inv_gaussian_rng(a_n / v_n, square(a_n));
//   }
// }
