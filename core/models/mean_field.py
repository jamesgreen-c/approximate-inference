import numpy as np
from scipy.special import expit


def MeanField(X, mu, sigma, pie, lambda0, maxsteps, eps: float = 1e-6) -> tuple:
    """
    Find lambda that maximise the free-energy
    lambda0: N x K
    X: N x D
    mu: D x K
    pie: 1 x K
    """

    N, D = X.shape
    K = mu.shape[1]

    # series of free-energy calculations
    F = []

    # add tolerance to pie for logs
    pie = np.clip(pie, 1e-16, 1-1e-16)

    # continue for a predefined number of steps
    for iteration in range(maxsteps):

        # calculate all lambda[n, i]
        for i in range(K):

            # first part of sigmoid
            first_term = np.repeat(np.log(pie[0, i] / (1 - pie[0, i])), N)

            # second term in sigmoid
            b = lambda0 @ mu.T
            c = ((0.5 - lambda0[:, i].reshape((N, 1))) @ mu[:, i].reshape((1, D)))
            second_term = ((X - b - c) @ mu[:, i]) / sigma**2

            lambda0[:, i] = expit(first_term + second_term)

        # update free energy
        free_energy = 0
        for n in range(N):

            lm_clipped = np.clip(lambda0[n, :], 1e-16, 1 - 1e-16)

            log_p_x = (- D * 0.5 * np.log(2 * np.pi)) - (D * np.log(sigma)) - ((
                    (X[n, :] - np.sum(mu * lm_clipped, axis=1)).T @ (X[n, :] - np.sum(mu * lm_clipped, axis=1))
            ) / (2 * sigma**2))
            log_p_s = (lm_clipped @ np.log(pie)[0]) + ((1 - lm_clipped) @ np.log(1 - pie)[0])
            entropy = (lm_clipped @ np.log(lm_clipped)) + ((1 - lm_clipped) @ np.log(1 - lm_clipped))

            free_energy += log_p_x + log_p_s + entropy

        # track free energies
        F.append(free_energy)
        # print(free_energy)

        if len(F) >= 2:
            if abs(free_energy - F[-2]) < eps:  # skip the free energy we just added
                # print(f"Converged on E-step iteration: {iteration}")
                break

    return lambda0, F[-1]


def m_step(X, ES, ESS):
    """
    mu, sigma, pie = MStep(X,ES,ESS)

    Inputs:
    -----------------
           X: shape (N, D) data matrix
          ES: shape (N, K) E_q[s]
         ESS: shape (K, K) sum over data points of E_q[ss'] (N, K, K)
                           if E_q[ss'] is provided, the sum over N is done for you.

    Outputs:
    --------
          mu: shape (D, K) matrix of means in p(y|{s_i},mu,sigma)
       sigma: shape (,)    standard deviation in same
         pie: shape (1, K) vector of parameters specifying generative distribution for s
    """
    N, D = X.shape
    if ES.shape[0] != N:
        raise TypeError('ES must have the same number of rows as X')
    K = ES.shape[1]
    if ESS.shape == (N, K, K):
        ESS = np.sum(ESS, axis=0)
    if ESS.shape != (K, K):
        raise TypeError('ESS must be square and have the same number of columns as ES')

    mu = np.dot(np.dot(np.linalg.inv(ESS), ES.T), X).T
    sigma = np.sqrt((np.trace(np.dot(X.T, X)) + np.trace(np.dot(np.dot(mu.T, mu), ESS))
                     - 2 * np.trace(np.dot(np.dot(ES.T, X), mu))) / (N * D))
    pie = np.mean(ES, axis=0, keepdims=True)

    return mu, sigma, pie


def LearnBinFactors(X, K, iterations, mu=None, sigma=None, pie=None) -> tuple:
    """
    Run EM on the data X.
    Search for K latent factors.
    Run a max of "iterations" steps

    lambda0: N x K
    X: N x D
    mu: D x K
    pie: 1 x K
    """

    N, D = X.shape

    # init params if none provided
    mu = np.random.randn(D, K) if mu is None else mu
    sigma = 1 if sigma is None else sigma
    pie = np.repeat(1 / K, K).reshape((1, K)) if pie is None else pie

    assert mu.shape == (D, K)
    assert pie.shape == (1, K)

    lambda0 = np.ones((N, K)) / K
    e_maxsteps = 200

    # store for F
    free_energies = []

    # run for max iterations
    for i in range(iterations):

        # E step
        ES, F = MeanField(X, mu, sigma, pie, lambda0, e_maxsteps)
        free_energies.append(F)
        # print(f"Iteration: {i}: Free Energy: {F}")

        # M step
        ESS = ES.T @ ES
        np.fill_diagonal(ESS, np.sum(ES, axis=0))  # correct E[s_i * s_i] = E[s_i]
        ESS = ESS + (np.eye(K) * 1e-6)
        mu, sigma, pie = m_step(X, ES, ESS)

        # check for F
        if i > 0:
            # if F < free_energies[-2]:
            #     print("WRONG! Guess Again")
            #     break
            if abs(F - free_energies[-2]) < 1e-3:
                print(f"EM Converged on iteration: {i}")
                break

    return mu, sigma, pie, free_energies
