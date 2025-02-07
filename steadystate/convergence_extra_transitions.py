import numpy as np
import matplotlib.pyplot as plt

from .transitions import P, Q, R, S, A

from .convergence_tools import convergence as f_convergence
from .dist_tools import get_stationary_dist

P0 = np.array([0.5, 0.5])  # 50% probability of starting in each state

pi_P = get_stationary_dist(P)
pi_Q = get_stationary_dist(Q)
pi_R = get_stationary_dist(R)
pi_S = get_stationary_dist(S)
pi_A = get_stationary_dist(A)

print(f"Stationary Dist of P: {pi_P}")
print(f"Stationary Dist of Q: {pi_Q}")
print(f"Stationary Dist of R: {pi_R}")
print(f"Stationary Dist of S: {pi_S}")
print(f"Stationary Dist of A: {pi_A}")

# Add new matrices to the analysis
additional_matrices = {
    "R (Slow Convergence)": (R, pi_R),
    "S (Fast Mixing)": (S, pi_S),
    "A (Absorbing)": (A, pi_A),
}

# Run error decay analysis for all matrices
max_steps = 500
results = {}

convergence_P = f_convergence(P, P0, pi_P, max_steps)
convergence_Q = f_convergence(Q, P0, pi_Q, max_steps)

for label, (M, pi) in additional_matrices.items():
    m_convergence = f_convergence(M, P0, pi, max_steps)
    results[label] = m_convergence


def plot_extra_transitions():
    plt.figure(figsize=(10, 6))

    # Plot original matrices P and Q
    plt.plot(convergence_P, label="P (Original)", linestyle='-')
    plt.plot(convergence_Q, label="Q (Original)", linestyle='-')

    # Plot new matrices
    for label, m_convergence in results.items():
        plt.plot(m_convergence, label=label)

    plt.xlabel("Steps (N)")
    plt.ylabel("L1 Error ||P^N P0 - π||")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.title("Convergence Speed for Different Transition Matrices")
    plt.show()

plot_extra_transitions()
