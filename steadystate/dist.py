import numpy as np


def get_stationary_dist(M):
    eigvals, eigvecs = np.linalg.eig(M.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)]  # Find eigenvector for eigenvalue 1
    stationary = np.real(stationary[:, 0])
    stationary /= stationary.sum()  # Normalize
    return stationary


def get_transition_dist(P, P0, N):
    PN = np.linalg.matrix_power(P, N)
    return P0 @ PN

