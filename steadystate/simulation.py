import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q, R, S, A

# Initial state distribution
P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

# Parameters
N = 50  # Number of steps

# Function to simulate a Markov chain
def simulate_markov_chain(P, P0, N):
    states = [np.random.choice([0, 1], p=P0)]  # Start at X0 based on P0
    for _ in range(N - 1):
        current_state = states[-1]
        next_state = np.random.choice([0, 1], p=P[current_state])
        states.append(next_state)
    return states

# Simulate Markov chains for P and Q
simulated_P = simulate_markov_chain(P, P0, N)
simulated_Q = simulate_markov_chain(Q, P0, N)
simulated_R = simulate_markov_chain(R, P0, N)
simulated_S = simulate_markov_chain(S, P0, N)
simulated_A = simulate_markov_chain(A, P0, N)

# Function to compute P^N (matrix exponentiation)
def compute_transition_distribution(P, P0, N):
    PN = np.linalg.matrix_power(P, N)  # Compute P^N
    return P0 @ PN  # Compute final state distribution

# Compute P^N * P0 and Q^N * P0
final_dist_P = compute_transition_distribution(P, P0, N)
final_dist_Q = compute_transition_distribution(Q, P0, N)
final_dist_R = compute_transition_distribution(R, P0, N)
final_dist_S = compute_transition_distribution(S, P0, N)
final_dist_A = compute_transition_distribution(A, P0, N)

# Print results
print(f"Final distribution using P: {final_dist_P}")
print(f"Final distribution using Q: {final_dist_Q}")
print(f"Final distribution using R: {final_dist_R}")
print(f"Final distribution using S: {final_dist_S}")
print(f"Final distribution using A: {final_dist_A}")

# Plot the simulated state sequences
def plot_simulated_chain():
    plt.figure(figsize=(10, 4))
    plt.plot(simulated_P, label="Markov Chain with P", linestyle='--')
    plt.plot(simulated_Q, label="Markov Chain with Q", linestyle='--')
    plt.plot(simulated_R, label="Markov Chain with R", linestyle='--')
    plt.plot(simulated_S, label="Markov Chain with S", linestyle='--')
    plt.plot(simulated_A, label="Markov Chain with A", linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.yticks([0, 1])
    plt.legend()
    plt.title("Markov Chain Simulation")
    plt.show()

plot_simulated_chain()
