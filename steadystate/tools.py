import numpy as np
from scipy.optimize import curve_fit

from .dist import get_transition_dist

def convergence_speed(P, P0, stationary_dist, threshold=1e-4, max_steps=5000):
    errors = []

    for N in range(1, max_steps + 1):
        transition_dist = get_transition_dist(P, P0, N)
        error = np.sum(np.abs(transition_dist - stationary_dist))  # Compute L1 norm
        errors.append(error)

        if error < threshold:
            return N, errors  # Return step count and error history
    return max_steps, errors  # If it doesn't converge in max_steps



