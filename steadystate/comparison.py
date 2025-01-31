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

# Function to compute stationary distribution
def compute_stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)  # Eigen decomposition
    stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Find eigenvector for eigenvalue 1
    stationary = stationary[:, 0].real
    stationary /= stationary.sum()  # Normalize
    return stationary

# Compute stationary distributions
pi_P = compute_stationary_distribution(P)
pi_Q = compute_stationary_distribution(Q)

# Function to compute error decay
def compute_error_decay(P, P0, pi, max_steps=1000):
    errors = []
    for N in range(1, max_steps + 1):
        PN_P0 = P0 @ np.linalg.matrix_power(P, N)  # Corrected propagation order
        error = np.sum(np.abs(PN_P0 - pi))  # Compute L1 norm
        errors.append(error)
    return np.array(errors)

# Define different initial distributions
initial_distributions = {
    "Uniform [0.5, 0.5]": np.array([0.5, 0.5]),
    "Start in State 0 [1, 0]": np.array([1, 0]),
    "Start in State 1 [0, 1]": np.array([0, 1]),
    "Close to π_P": pi_P,
}

# Parameters
max_steps = 500

# Run for each P0 choice
results = {}
for label, P0 in initial_distributions.items():
    errors_P = compute_error_decay(P, P0, pi_P, max_steps)
    errors_Q = compute_error_decay(Q, P0, pi_Q, max_steps)
    results[label] = (errors_P, errors_Q)

# Plot results
plt.figure(figsize=(10, 6))

for label, (errors_P, errors_Q) in results.items():
    plt.plot(errors_P, label=f"P - {label}", linestyle='-')
    plt.plot(errors_Q, label=f"Q - {label}", linestyle='--')

plt.xlabel("Steps (N)")
plt.ylabel("L1 Error ||P^N P0 - π||")
plt.yscale("log")  # Log scale for better visualization
plt.legend()
plt.title("Effect of Different Initial Distributions on Convergence")
plt.show()
