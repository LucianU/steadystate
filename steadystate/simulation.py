import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q, R, S, A
from .dist_tools import get_stationary_dist

# Initial state distribution
P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

# Parameters
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
simulated_R = simulate_markov_chain(R, P0, N)
simulated_S = simulate_markov_chain(S, P0, N)
simulated_A = simulate_markov_chain(A, P0, N)


stationary_dist_P = get_stationary_dist(P)
stationary_dist_Q = get_stationary_dist(Q)
stationary_dist_R = get_stationary_dist(R)
stationary_dist_S = get_stationary_dist(S)
stationary_dist_A = get_stationary_dist(A)


errors_P = [compute_error(simulated_P[:i], stationary_dist_P) for i in range(1, N+1)]
errors_Q = [compute_error(simulated_Q[:i], stationary_dist_Q) for i in range(1, N+1)]
errors_R = [compute_error(simulated_R[:i], stationary_dist_R) for i in range(1, N+1)]
errors_S = [compute_error(simulated_S[:i], stationary_dist_S) for i in range(1, N+1)]
errors_A = [compute_error(simulated_A[:i], stationary_dist_A) for i in range(1, N+1)]


def plot_simulation_error():
    plt.figure(figsize=(8, 5))
    plt.plot(errors_P, label="Simulated errors for P")
    plt.plot(errors_Q, label="Simulated errors for Q")
    plt.plot(errors_R, label="Simulated errors for R")
    plt.plot(errors_S, label="Simulated errors for S")
    plt.plot(errors_A, label="Simulated errors for A")
    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Steps")
    plt.ylabel("L1 Error ||Distribution - π||")
    plt.legend()
    plt.title("Error of Convergent Markov Chain (Decreasing Error)")
    plt.show()

plot_simulation_error()

