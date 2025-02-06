import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q
from .tools import decay_rate, eigen_speed, get_stationary_dist, error_decay

# Initial state distribution
P0 = np.array([0.5, 0.5])

# Compute stationary distributions
pi_P = get_stationary_dist(P)
pi_Q = get_stationary_dist(Q)

# Measure error decay
max_steps = 500

N_vals = np.arange(1, max_steps + 1)

errors_P = error_decay(P, P0, pi_P, max_steps)
errors_Q = error_decay(Q, P0, pi_Q, max_steps)

r_P = decay_rate(errors_P, N_vals)
r_Q = decay_rate(errors_Q, N_vals)

print(f"Convergence rate (r) for P: {1/abs(r_P)}")
print(f"Convergence rate (r) for Q: {1/abs(r_Q)}")

speed_P = eigen_speed(P)
speed_Q = eigen_speed(Q)

print(f"Speed estimate for P (1 - |λ2|): {speed_P}")
print(f"Speed estimate for Q (1 - |λ2|): {speed_Q}")

# Plot error decay
def speed_plot():
    plt.figure(figsize=(8, 5))
    plt.plot(N_vals, errors_P, label=f"Error for P (r={r_P:.3f})", marker='o')
    plt.plot(N_vals, errors_Q, label=f"Error for Q (r={r_Q:.3f})", marker='x', linestyle='--')
    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed of Markov Chains")
    plt.show()

speed_plot()
