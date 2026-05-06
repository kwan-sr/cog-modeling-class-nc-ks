functions {
  real wald_lpdf(real y, real v, real a) {
    if (y <= 0) return negative_infinity();
    return log(a) - 0.5 * log(2 * pi()) - 1.5 * log(y)
           - square(a - v * y) / (2 * y);
  }

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
  int<lower=1> N;
  int<lower=1> N_subj;
  int<lower=1> N_char;
  array[N] int<lower=1, upper=2> hand;       // 1 = Dominant, 2 = Non-dominant
  array[N] int<lower=1, upper=2> speed;      // 1 = fast, 2 = slow
  array[N] int<lower=1, upper=N_subj> subj;
  array[N] int<lower=1, upper=N_char> char_idx;
  vector<lower=0>[N] y;
}

parameters {
  vector<lower=0>[2] mu_v;        // drift by Hand only
  vector<lower=0>[2] mu_a;        // threshold by Speed only
  vector[N_subj] z_v_raw;
  vector[N_subj] z_a_raw;
  vector[N_char] z_c_raw;
  real<lower=0> sigma_v;
  real<lower=0> sigma_a;
  real<lower=0> sigma_c;
}

transformed parameters {
  vector[N_subj] z_v = z_v_raw * sigma_v;
  vector[N_subj] z_a = z_a_raw * sigma_a;
  vector[N_char] z_c = z_c_raw * sigma_c;
}

model {
  mu_v ~ gamma(3, 1);
  mu_a ~ gamma(3, 1);
  sigma_v ~ normal(0, 0.5);
  sigma_a ~ normal(0, 0.5);
  sigma_c ~ normal(0, 0.5);
  z_v_raw ~ std_normal();
  z_a_raw ~ std_normal();
  z_c_raw ~ std_normal();

  for (n in 1:N) {
    real v_n = mu_v[hand[n]]  * exp(z_v[subj[n]] + z_c[char_idx[n]]);
    real a_n = mu_a[speed[n]] * exp(z_a[subj[n]]);
    y[n] ~ wald(v_n, a_n);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    real v_n = mu_v[hand[n]]  * exp(z_v[subj[n]] + z_c[char_idx[n]]);
    real a_n = mu_a[speed[n]] * exp(z_a[subj[n]]);
    log_lik[n] = wald_lpdf(y[n] | v_n, a_n);
    y_rep[n] = inv_gaussian_rng(a_n / v_n, square(a_n));
  }
}
