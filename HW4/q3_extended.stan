data {
    int<lower=0> N;                     // Number of training data entries
    int<lower=1> J;                     // Number of subjects
    array[N] int<lower=0, upper=1> y;   // decision is only 1 or 0 so should be int
    vector[N] attr;   // normalized scores are real numbers between 0 and 1
    vector[N] intel;
    vector[N] fun;
    array[N] int mask;                  //Mask to help indexing each participant
}

parameters {
    // Model assumes different priors for each participant
    // This makes sense because each participant is probably going to value each attribute differently
    vector[J] a;
    vector[J] b;
    vector[J] c;
    vector[J] d;

}

model {
    // Prior
    a ~ normal(0,2); // Sampling J different priors (one for each participant)
    b ~ normal(0,2);
    c ~ normal(0,2);
    d ~ normal(0,2);

    // Likelihood
    for (n in 1:N) {
        // this version of bernoulli first applies sigmoid function to convert the regression result into probability 
        y[n] ~ bernoulli_logit(a[mask[n]] * attr[n] + b[mask[n]] * intel[n] + c[mask[n]] * fun[n] + d[mask[n]]);
    }   
}