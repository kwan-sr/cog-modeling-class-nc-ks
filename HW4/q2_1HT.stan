data {
  int<lower=0> N_old;
  int<lower=0> N_new;
  int<lower=0> y_old_old;   // old items judged "old" (hits)
  int<lower=0> y_new_old;   // new items judged "old" (false alarms)
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

model {
  // Priors
  d ~ beta(1, 1);
  g ~ beta(1, 1);

  // Likelihood
  y_old_old ~ binomial(N_old, d + (1 - d) * g);
  y_new_old ~ binomial(N_new, g);
}