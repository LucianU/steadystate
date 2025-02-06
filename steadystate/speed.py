import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q
from .convergence import convergence_rate, convergence, eigen_speed
from .dist import get_stationary_dist

# Initial state distribution
P0 = np.array([0.5, 0.5])

# Compute stationary distributions
stationary_dist_P = get_stationary_dist(P)
stationary_dist_Q = get_stationary_dist(Q)

# Measure error decay
max_steps = 500

N_vals = np.arange(1, max_steps + 1)

convergence_P = convergence(P, P0, stationary_dist_P, max_steps)
convergence_Q = convergence(Q, P0, stationary_dist_Q, max_steps)

convergence_rate_P = convergence_rate(convergence_P, N_vals)
convergence_rate_Q = convergence_rate(convergence_Q, N_vals)

print(f"Convergence rate (r) for P: {convergence_rate_P}")
print(f"Convergence rate (r) for Q: {convergence_rate_Q}")

speed_P = eigen_speed(P)
speed_Q = eigen_speed(Q)

print(f"Eigenvalue speed estimate for P (1 - |λ2|): {speed_P}")
print(f"Eigenvalue speed estimate for Q (1 - |λ2|): {speed_Q}")

# Plot error decay
def speed_plot():
    plt.figure(figsize=(8, 5))
    plt.plot(N_vals, convergence_P, label=f"Error for P (r={convergence_rate_P:.3f})")
    plt.plot(N_vals, convergence_Q, label=f"Error for Q (r={convergence_rate_Q:.3f})", linestyle='--')
    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed of Markov Chains")
    plt.show()

speed_plot()
