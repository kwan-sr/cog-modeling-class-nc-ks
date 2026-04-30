data {
    int<lower=1> N;
    array[N] real<lower=0> y;
    array[N] int<lower=0, upper=1> c;
}

parameters {
    real<lower=0> v;
    real<lower=0> a;
    real<lower=0, upper=1> beta;
    real<lower=0, upper=min(y) - 1e-6> tau; //is bounded between 0 and the lowest RT (bc we can't have non-decision time that's more than any RT)
}

model {
    // Priors
    v ~ gamma(3, 1);
    a ~ gamma(3,1);
    beta ~ beta(2,2);
    tau ~ gamma(2,1);

    // Likelihood
    for (n in 1:N) {
        if (c[n] == 1) {
            y[n] ~ wiener(a, tau, beta, v);
        }
        else {
            y[n] ~ wiener(a, tau, 1-beta, -v);
        }
    }
}