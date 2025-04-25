import numpy as np
from scipy.special import expit
np.random.seed(0)


def calculate_boltzmann_params(mu, sigma, X, pie):
    """
    Calculate new W and b for the true factors
    """

    N = X.shape[0]
    K = pie.shape[1]

    # calc W
    W = - (mu.T @ mu) / (2 * sigma ** 2)

    # calc b
    b = np.zeros((N, K))
    for i in range(K):
        term_1 = - (X @ mu[:, i]) / sigma ** 2
        term_2 = (mu[:, i] @ mu[:, i]) / (2 * sigma ** 2)
        term_3 = - (np.log(pie[0, i]) - np.log(1 - pie[0, i]))
        b_i = term_1 + term_2 + term_3
        b[:, i] = b_i

    return b, W


class LoopyBP:
    """
    class stores the current factor beliefs for f and g via the natural params
    methods:
        - update_f_i updates the belief about f
        - update_g_ij updates the belief about g_ij
        - loop runs a single loop around the graph BP.
    """

    def __init__(self, X, K):

        self.N, self.D = X.shape
        self.K = K

        self.X = X
        self.eta = np.random.randn(self.N, K, K)

    def update_f_i(self, b: np.ndarray, i: int):
        """
        Calculate natural params for f_i(s_i)
        n_ii: (N, ) array (for all N observations)
        """
        self.eta[:, i, i] = - b[:, i].reshape(-1)

    def update_g_ij(self, W, i, j, clip: float = 1e-6):
        """
        Calculate natural params for g_ij(s_i, s_j)
        Need to sum incoming neighbour messages (is every other node due to nature of model)

        n_ij: (N, ) array (for all N observations)
        """

        neighbourhood = (np.sum(self.eta[:, :j, i], axis=1) + np.sum(self.eta[:, j + 1:, i], axis=1)).reshape(-1)
        message = self.eta[:, i, i] + neighbourhood
        self.eta[:, i, j] = np.log(1 + np.exp(W[i, j] + message) + clip) - np.log(1 + np.exp(message) + clip)

    def loop(self, b: np.ndarray, W: np.ndarray, iterations: int = 1):

        for iteration in range(iterations):
            for i in range(self.K):
                self.update_f_i(b, i)

                for j in range(i):
                    self.update_g_ij(W, i, j)
                    self.update_g_ij(W, j, i)


def calculate_free_energy(X, N, lambda0, D, sigma, mu, pie):
    """ Define function to calc free energy """

    # update free energy
    free_energy = 0
    for n in range(N):
        lm_clipped = np.clip(lambda0[n, :], 1e-16, 1 - 1e-16)

        log_p_x = (- D * 0.5 * np.log(2 * np.pi)) - (D * np.log(sigma)) - ((
               (X[n, :] - np.sum(mu * lm_clipped, axis=1)).T @ (X[n, :] - np.sum(mu * lm_clipped, axis=1))
        ) / (2 * sigma ** 2))
        log_p_s = (lm_clipped @ np.log(pie)[0]) + ((1 - lm_clipped) @ np.log(1 - pie)[0])
        entropy = (lm_clipped @ np.log(lm_clipped)) + ((1 - lm_clipped) @ np.log(1 - lm_clipped))

        free_energy += log_p_x + log_p_s + entropy

    return free_energy


def e_step(X, mu, sigma, pie, lambda0, maxsteps, loopy_bp: LoopyBP, eps: float = 1e-6):
    """
    Find lambda that maximise the free-energy
    lambda0: N x K
    X: N x D
    mu: D x K
    pie: 1 x K
    """

    N, D = X.shape
    K = mu.shape[1]
    F = []
    pie = np.clip(pie, 1e-16, 1-1e-16)  # add tolerance to pie for logs

    # calculate b and W
    b, W = calculate_boltzmann_params(mu, sigma, X, pie)

    # calculate lambda matrix and free energy
    loopy_bp.loop(b=b, W=W, iterations=5)
    eta = loopy_bp.eta
    lambda0 = expit(eta.sum(axis=1))

    # track free energies
    F.append(calculate_free_energy(X, N, lambda0, D, sigma, mu, pie))
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

    # init loopy BP
    loopy_bp = LoopyBP(X, K)

    # run for max iterations
    for i in range(iterations):

        # E step
        ES, F = e_step(X, mu, sigma, pie, lambda0, e_maxsteps, loopy_bp)
        free_energies.append(F)
        # print(f"Iteration: {i}: Free Energy: {F}")

        # M step
        ESS = ES.T @ ES
        np.fill_diagonal(ESS, np.sum(ES, axis=0))  # correct E[s_i * s_i] = E[s_i]
        ESS = ESS + (np.eye(K) * 1e-6)
        mu, sigma, pie = m_step(X, ES, ESS)

        # check for F
        if i > 0:
            if abs(F - free_energies[-2]) < 1e-3:
                print(f"EM Converged on iteration: {i}")
                break

    return mu, sigma, pie, free_energies

