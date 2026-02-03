
def simulate_seir(parameters, init_conditions, days=51):
    """TODO"""

    # Extract parameters and initial conditions (Your code here)
    beta, sigma, gamma = parameters
    S0, E0, I0, R0 = init_conditions
    N = S0 + E0 + I0 + R0
    S = [S0] + [None] * (days-1)
    E = [E0] + [None] * (days-1)
    I = [I0] + [None] * (days-1)
    R = [R0] + [None] * (days-1)
    
    # For each day, perform SEIR update
    for t in range(1, days):

        # Compute new cases and update equations
        E_new = (beta*S[t-1]*I[t-1]) / N
        I_new = sigma*E[t-1]
        R_new = gamma*I[t-1]
      
        # Update equations
        S[t] = S[t-1] - E_new
        E[t] = E[t-1] + E_new - I_new
        I[t] = I[t-1] + I_new - R_new
        R[t] = R[t-1] + R_new

    return (S, E, I, R)
