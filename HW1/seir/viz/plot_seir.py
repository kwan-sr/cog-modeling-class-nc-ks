import matplotlib.pyplot as plt


def plot_results(infected):
    """Plots the time series of infected cases."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(infected, color='#AA0000', linestyle='dashed', marker='o')
    ax.set_xlabel('Day', fontsize=16)
    ax.set_ylabel('Number of Infected Cases', fontsize=16)
    ax.set_title('Simulated Oubreak', fontsize=20)
    ax.grid(alpha=0.2)
    return fig
