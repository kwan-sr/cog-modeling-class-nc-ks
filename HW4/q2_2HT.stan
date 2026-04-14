data {
  int<lower=0> N_old;
  int<lower=0> N_new;
  int<lower=0> y_old_old;
  int<lower=0> y_new_old;
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

model {
  d ~ beta(1,1);
  g ~ beta(1,1);

  // Old items
  y_old_old ~ binomial(N_old, d + (1 - d) * g);

  // New items
  y_new_old ~ binomial(N_new, (1 - d) * g);
}