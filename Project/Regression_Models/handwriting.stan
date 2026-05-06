data {
    int<lower=1> N;
    int<lower=1> J; // participants
    int<lower=1> K; // characters

    array[N] int<lower=1, upper=J> participant;
    array[N] int<lower=1, upper=K> char;

    vector[N] hand;
    vector[N] speed;
    vector[N] interaction;

    vector[N] y;
}

parameters {
    real mu_participant;
    real<lower=0> sigma_participant;

    vector[J] alpha_participant;
    vector[K] alpha_char;

    real beta_hand;
    real beta_speed;
    real beta_interaction;

    real<lower=0> sigma;
}

model {
    // priors
    mu_participant ~ normal(0, 1);
    sigma_participant ~ exponential(1);

    alpha_participant ~ normal(mu_participant, sigma_participant);
    alpha_char ~ normal(0, 1);

    beta_hand ~ normal(0, 1);
    beta_speed ~ normal(0, 1);
    beta_interaction ~ normal(0, 1);

    sigma ~ exponential(1);

    // likelihood
    for (n in 1:N) {
        y[n] ~ normal(
            alpha_participant[participant[n]]
            + alpha_char[char[n]]
            + beta_hand * hand[n]
            + beta_speed * speed[n]
            + beta_interaction * interaction[n],
            sigma
        );
    }
}