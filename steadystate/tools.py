import numpy as np
from scipy.optimize import curve_fit


def get_stationary_dist(M):
    eigvals, eigvecs = np.linalg.eig(M.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Find eigenvector for eigenvalue 1
    stationary = np.real(stationary[:, 0])
    stationary /= stationary.sum()  # Normalize
    return stationary


def get_transition_dist(P, P0, N):
    PN = np.linalg.matrix_power(P, N)
    return P0 @ PN


def convergence_speed(P, P0, stationary_dist, threshold=1e-4, max_steps=5000):
    errors = []

    for N in range(1, max_steps + 1):
        transition_dist = get_transition_dist(P, P0, N)
        error = np.sum(np.abs(transition_dist - stationary_dist))  # Compute L1 norm
        errors.append(error)

        if error < threshold:
            return N, errors  # Return step count and error history
    return max_steps, errors  # If it doesn't converge in max_steps


def eigen_speed(M):
    """Compute second-largest eigenvalues for speed comparison"""
    eigenvalues_M = np.linalg.eigvals(M)
    lambda2_M = sorted(np.abs(eigenvalues_M))[-2]
    speed_M = 1 - lambda2_M
    return speed_M


def error_decay(M, P0, stationary_dist, max_steps=1000):
    stationary_dist = get_stationary_dist(M)
    errors = []

    for N in range(1, max_steps + 1):
        transition_dist = get_transition_dist(M, P0, N)
        error = np.sum(np.abs(transition_dist - stationary_dist))  # Compute L1 norm
        errors.append(error)

    return np.array(errors)


def decay_rate(errors_M, N_vals):
    popt_M, _ = curve_fit(exp_decay, N_vals, errors_M, p0=(1, 0.1))  # Fit
    r_M = popt_M[1]  # Decay rate
    return r_M


def exp_decay(N, A, r):
    return A * np.exp(-r * N)

