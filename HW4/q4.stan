data {
    int<lower=0> N;                     
    array[N] int<lower=0, upper=1> y;   
    vector[N] attr;
    vector[N] intel;
    vector[N] fun;

    int<lower=0> M;                     // number of test cases
    vector[M] attr_test;
    vector[M] intel_test;
    vector[M] fun_test;
}

parameters {
    real a;
    real b;
    real c;
    real d;
}

model {
    // Priors
    a ~ normal(0,2);
    b ~ normal(0,2);
    c ~ normal(0,2);
    d ~ normal(0,2);

    // Likelihood on training data
    for (n in 1:N) {
        y[n] ~ bernoulli_logit(a * attr[n] + b * intel[n] + c * fun[n] + d);
    }   
}

generated quantities {
    vector[M] p_test;                // posterior predictive probabilities
    array[M] int y_test_rep;         // optional replicated binary outcomes

    for (m in 1:M) {
        p_test[m] = inv_logit(a * attr_test[m] + b * intel_test[m] + c * fun_test[m] + d);
        y_test_rep[m] = bernoulli_rng(p_test[m]);
    }
}