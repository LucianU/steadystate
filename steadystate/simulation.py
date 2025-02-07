import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q
from .dist_tools import get_stationary_dist

# Initial state distribution
P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

N = 1000  # Number of steps

# Function to simulate a Markov chain
def simulate_markov_chain(P, P0, N):
    states = [np.random.choice([0, 1], p=P0)]  # Start at X0 based on P0
    for _ in range(N - 1):
        current_state = states[-1]
        next_state = np.random.choice([0, 1], p=P[current_state])
        states.append(next_state)
    return states

# Function to compute L1 error between the current distribution and stationary distribution
def compute_error(states, stationary_dist):
    state_counts = np.bincount(states, minlength=2) / len(states)
    return np.sum(np.abs(state_counts - stationary_dist))

# Simulate Markov chains for P and Q
simulated_P = simulate_markov_chain(P, P0, N)
simulated_Q = simulate_markov_chain(Q, P0, N)

stationary_dist_P = get_stationary_dist(P)
stationary_dist_Q = get_stationary_dist(Q)


errors_P = [compute_error(simulated_P[:i], stationary_dist_P) for i in range(1, N+1)]
errors_Q = [compute_error(simulated_Q[:i], stationary_dist_Q) for i in range(1, N+1)]


def plot_simulation(log=False):
    plt.figure(figsize=(8, 5))
    plt.plot(errors_P, label="Simulated errors for P")
    plt.plot(errors_Q, label="Simulated errors for Q")
    if log:
        plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Steps")
    plt.ylabel("L1 Error ||Distribution - π||")
    plt.legend()
    title = "Error of Convergent Markov Chain"
    if log:
        title = f"{title} (Decreasing)"
    plt.title(title)
    plt.show()

plot_simulation()

