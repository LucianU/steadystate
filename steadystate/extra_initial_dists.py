"""
Compare how different initial distributions evolve under the same transition matrices.
"""

import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q

from .convergence_tools import convergence
from .dist_tools import get_stationary_dist


# Compute stationary distributions
pi_P = get_stationary_dist(P)
pi_Q = get_stationary_dist(Q)

# Define different initial distributions
initial_distributions = {
    "Start in either State [0.5, 0.5]": np.array([0.5, 0.5]),
    "Start in State 0 [1, 0]": np.array([1, 0]),
    "Start in State 1 [0, 1]": np.array([0, 1]),
    "Close to π_P": pi_P,
}

# Parameters
max_steps = 500

# Run for each P0 choice
results = {}
for label, P0 in initial_distributions.items():
    convergence_P = convergence(P, P0, pi_P, max_steps)
    convergence_Q = convergence(Q, P0, pi_Q, max_steps)
    results[label] = (convergence_P, convergence_Q)


def plot_extra_initial_dist():
    plt.figure(figsize=(10, 6))

    for label, (convergence_P, convergence_Q) in results.items():
        plt.plot(convergence_P, label=f"P - {label}", linestyle='-')
        plt.plot(convergence_Q, label=f"Q - {label}", linestyle='--')

    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Effect of Different Initial Distributions on Convergence")
    plt.show()

plot_extra_initial_dist()
