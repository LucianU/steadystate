import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q, R, S, A
from .tools import  get_stationary_dist, convergence_speed

# Initial state distribution
P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

pi_P = get_stationary_dist(P)
pi_Q = get_stationary_dist(Q)
pi_R = get_stationary_dist(R)
pi_S = get_stationary_dist(S)
pi_A = get_stationary_dist(A)

print(f"Stationary distribution for P: {pi_P}")
print(f"Stationary distribution for Q: {pi_Q}")
print(f"Stationary distribution for R: {pi_R}")
print(f"Stationary distribution for S: {pi_S}")
print(f"Stationary distribution for A: {pi_A}")

# Compute convergence speed for P and Q
N_P, errors_P = convergence_speed(P, P0, pi_P)
N_Q, errors_Q = convergence_speed(Q, P0, pi_Q)
N_R, errors_R = convergence_speed(R, P0, pi_R)
N_S, errors_S = convergence_speed(S, P0, pi_S)
N_A, errors_A = convergence_speed(A, P0, pi_A)

print(f"Convergence steps for P: {N_P}")
print(f"Convergence steps for Q: {N_Q}")
print(f"Convergence steps for R: {N_R}")
print(f"Convergence steps for S: {N_S}")
print(f"Convergence steps for A: {N_A}")

# Plot convergence error over time
def plot_convergence_rate():
    plt.figure(figsize=(10, 6))
    plt.plot(errors_P, label="Error for P", linestyle='--')
    plt.plot(errors_Q, label="Error for Q", linestyle='--')
    plt.plot(errors_R, label="Error for R", linestyle='--')
    plt.plot(errors_S, label="Error for S", linestyle='--')
    plt.plot(errors_A, label="Error for A", linestyle='--')
    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed of Markov Chains")
    plt.show()

plot_convergence_rate()
