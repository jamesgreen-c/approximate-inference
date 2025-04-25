import numpy as np


def covariance_kernel(s, t, phi, psi, tau, sigma, eta, theta):
    """
    Calculate covariance kernel for the given kernel function in question 2d
    Use Form k(s, t) = theta**2(a + b) + c
    """
    a = np.exp(-2 * (np.sin(np.pi * (s - t) / tau) ** 2) / sigma ** 2)
    b = phi ** 2 * np.exp(-0.5 * ((s - t) ** 2) / (eta ** 2))
    c = psi ** 2 if s == t else 0  # delta function (delta == 1 if s == t otherwise 0)

    return (theta ** 2 * (a + b)) + c
