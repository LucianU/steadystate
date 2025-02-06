import numpy as np
import matplotlib.pyplot as plt

from .tools import get_stationary_dist, error_decay
from .transitions import P, Q, R, S, A

P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

# Compute stationary distributions for new matrices
pi_P = get_stationary_dist(P)
pi_Q = get_stationary_dist(Q)
pi_R = get_stationary_dist(R)
pi_S = get_stationary_dist(S)
pi_A = get_stationary_dist(A)

# Add new matrices to the analysis
additional_matrices = {
    "R (Slow Convergence)": (R, pi_R),
    "S (Fast Mixing)": (S, pi_S),
    "A (Absorbing)": (A, pi_A),
}

# Run error decay analysis for all matrices
max_steps = 500
results = {}

errors_P = error_decay(P, P0, pi_P, max_steps)
errors_Q = error_decay(Q, P0, pi_Q, max_steps)

for label, (matrix, pi) in additional_matrices.items():
    errors = error_decay(matrix, P0, pi, max_steps)
    results[label] = errors

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot original matrices P and Q
plt.plot(errors_P, label="P (Original)", linestyle='-')
plt.plot(errors_Q, label="Q (Original)", linestyle='--')

# Plot new matrices
for label, errors in results.items():
    plt.plot(errors, label=label)

plt.xlabel("Steps (N)")
plt.ylabel("L1 Error ||P^N P0 - π||")
plt.yscale("log")  # Log scale for better visualization
plt.legend()
plt.title("Convergence Speed for Different Transition Matrices")
plt.show()
