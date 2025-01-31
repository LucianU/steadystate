import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
P0 = np.array([0.5, 0.5])

# Function to compute stationary distribution
def compute_stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)  # Eigen decomposition
    stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Get eigenvector for eigenvalue 1
    stationary = stationary[:, 0].real
    stationary /= stationary.sum()  # Normalize
    return stationary

# Compute stationary distributions
pi_P = compute_stationary_distribution(P)
pi_Q = compute_stationary_distribution(Q)

# Function to compute error over steps
def compute_error_decay(P, P0, pi, max_steps=1000):
    errors = []
    for N in range(1, max_steps + 1):
        PN_P0 = np.linalg.matrix_power(P.T, N) @ P0  # Compute P^N * P0
        error = np.sum(np.abs(PN_P0 - pi))  # Compute L1 norm
        errors.append(error)
    return np.array(errors)

# Measure error decay
max_steps = 500
errors_P = compute_error_decay(P, P0, pi_P, max_steps)
errors_Q = compute_error_decay(Q, P0, pi_Q, max_steps)

# Function to fit exponential decay
def exp_decay(N, A, r):
    return A * np.exp(-r * N)

# Fit error curves to extract convergence rate
N_vals = np.arange(1, max_steps + 1)
popt_P, _ = curve_fit(exp_decay, N_vals, errors_P, p0=(1, 0.1))  # Fit for P
popt_Q, _ = curve_fit(exp_decay, N_vals, errors_Q, p0=(1, 0.1))  # Fit for Q

r_P = popt_P[1]  # Decay rate for P
r_Q = popt_Q[1]  # Decay rate for Q

print(f"Convergence rate (r) for P: {1/abs(r_P)}")
print(f"Convergence rate (r) for Q: {1/abs(r_Q)}")

# Compute second-largest eigenvalues for speed comparison
eigenvalues_P = np.linalg.eigvals(P)
eigenvalues_Q = np.linalg.eigvals(Q)

lambda2_P = sorted(np.abs(eigenvalues_P))[-2]
lambda2_Q = sorted(np.abs(eigenvalues_Q))[-2]

speed_P = 1 - lambda2_P
speed_Q = 1 - lambda2_Q

print(f"Speed estimate for P (1 - |λ2|): {speed_P}")
print(f"Speed estimate for Q (1 - |λ2|): {speed_Q}")

# Plot error decay
def plot():
    plt.figure(figsize=(8, 5))
    plt.plot(N_vals, errors_P, label=f"Error for P (r={r_P:.3f})", marker='o')
    plt.plot(N_vals, errors_Q, label=f"Error for Q (r={r_Q:.3f})", marker='x', linestyle='--')
    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed of Markov Chains")
    plt.show()
