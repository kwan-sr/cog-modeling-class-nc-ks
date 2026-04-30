data {
    int<lower=1> N_fast;
    int<lower=1> N_slow;
    array[N_fast] real<lower=0> y_fast;
    array[N_slow] real<lower=0> y_slow;
}

parameters {
    real<lower=0> v;
    real<lower=0> a_fast;
    real<lower=0> a_slow;
    real<lower=0, upper=1> beta;
    real<lower=0, upper=min(y_fast) - 1e-6> tau; //is bounded between 0 and the lowest RT (bc we can't have non-decision time that's more than any RT)
    real<lower=0, upper=min(y_slow) - tau - 1e-6> T;
}

model {
    // Priors
    v ~ gamma(3, 1);
    a_fast ~ gamma(3,1);
    a_slow ~ gamma(3,1);
    beta ~ beta(2,2);
    tau ~ gamma(2,1);
    T ~ gamma(2,1);

    // Likelihood
    for (n in 1:N_fast) {
            y_fast[n] ~ wiener(a_fast, tau, beta, v);
    }  
    for (n in 1:N_slow) {
            y_slow[n] ~ wiener(a_slow, tau+T, beta, v);
    }
}