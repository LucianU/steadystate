import numpy as np
import matplotlib.pyplot as plt

# Define transition matrices P and Q
P = np.array([
    [0.9, 0.1],
    [0.7, 0.3]
])

Q = np.array([
    [0.95, 0.05],
    [0.01, 0.99]
])

# Initial state distribution
P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

# Function to simulate a Markov chain
def simulate_markov_chain(P, P0, N):
    states = [np.random.choice([0, 1], p=P0)]  # Start at X0 based on P0
    for _ in range(N - 1):
        current_state = states[-1]
        next_state = np.random.choice([0, 1], p=P[current_state])
        states.append(next_state)
    return states

def compute_stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)  # Get eigenvectors of P^T
    stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Find eigenvector for eigenvalue 1
    stationary = stationary[:, 0].real  # Extract real part
    stationary /= stationary.sum()  # Normalize
    return stationary

pi_P = compute_stationary_distribution(P)
pi_Q = compute_stationary_distribution(Q)

print(f"Stationary distribution for P: {pi_P}")
print(f"Stationary distribution for Q: {pi_Q}")

# Function to compute convergence speed
def compute_convergence_speed(P, P0, pi, threshold=1e-4, max_steps=5000):
    errors = []
    for N in range(1, max_steps + 1):
        PN_P0 = np.linalg.matrix_power(P, N) @ P0  # Compute P^N * P0
        error = np.sum(np.abs(PN_P0 - pi))  # Compute L1 norm
        errors.append(error)
        if error < threshold:
            return N, errors  # Return step count and error history
    return max_steps, errors  # If it doesn't converge in max_steps

# Compute convergence speed for P and Q
N_P, errors_P = compute_convergence_speed(P, P0, pi_P)
N_Q, errors_Q = compute_convergence_speed(Q, P0, pi_Q)

print(f"Convergence steps for P: {N_P}")
print(f"Convergence steps for Q: {N_Q}")
print("Last 10 errors for P:", errors_P[-10:])
print("Last 10 errors for Q:", errors_Q[-10:])

# Plot convergence error over time
def plot_convergence_error():
    plt.figure(figsize=(8, 5))
    plt.plot(errors_P, label="Error for P", marker='o')
    plt.plot(errors_Q, label="Error for Q", marker='x', linestyle='--')
    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed of Markov Chains")
    plt.show()

plot_convergence_error()

# Parameters
N = 50  # Number of steps

# Simulate Markov chains for P and Q
simulated_P = simulate_markov_chain(P, P0, N)
simulated_Q = simulate_markov_chain(Q, P0, N)

# Function to compute P^N (matrix exponentiation)
def compute_transition_distribution(P, P0, N):
    PN = np.linalg.matrix_power(P, N)  # Compute P^N
    return P0 @ PN  # Compute final state distribution

# Compute P^N * P0 and Q^N * P0
final_dist_P = compute_transition_distribution(P, P0, N)
final_dist_Q = compute_transition_distribution(Q, P0, N)

# Print results
print(f"Final distribution using P: {final_dist_P}")
print(f"Final distribution using Q: {final_dist_Q}")

# Plot the simulated state sequences
plt.figure(figsize=(10, 4))
plt.plot(simulated_P, label="Markov Chain with P", marker='o', linestyle='-')
plt.plot(simulated_Q, label="Markov Chain with Q", marker='x', linestyle='--')
plt.xlabel("Time step")
plt.ylabel("State")
plt.yticks([0, 1])
plt.legend()
plt.title("Markov Chain Simulation")
plt.show()

