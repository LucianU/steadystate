import numpy as np
from scipy.optimize import curve_fit

from .dist_tools import get_transition_dist


def convergence(M, P0, stationary_dist, max_steps=1000):
    errors = []

    for N in range(1, max_steps + 1):
        transition_dist = get_transition_dist(M, P0, N)
        error = np.sum(np.abs(transition_dist - stationary_dist))  # Compute L1 norm
        errors.append(error)

    return np.array(errors)


def convergence_rate(errors_M, N_vals):
    popt_M, _ = curve_fit(exp_decay, N_vals, errors_M, p0=(1, 0.1))  # Fit
    r_M = popt_M[1]  # Decay rate
    return r_M


def exp_decay(N, A, r):
    return A * np.exp(-r * N)


def eigen_speed(M):
    """Compute second-largest eigenvalues for speed comparison"""
    eigenvalues_M = np.linalg.eigvals(M)
    lambda2_M = sorted(np.abs(eigenvalues_M))[-2]
    speed_M = 1 - lambda2_M
    return speed_M

