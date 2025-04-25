import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


def zero_mean(x: np.ndarray):
    return np.zeros_like(x)


class GP:

    def __init__(self,
        mean: Callable,
        kernel: Callable,
        kernel_params: dict,
    ):

        self.mean = mean
        self.kernel = kernel
        self.kernel_params = kernel_params

    def sample(self, x: np.ndarray, K):
        return np.random.multivariate_normal(self.mean(x), K)

    def make_gram_matrix(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:

        n1, n2 = len(x1), len(x2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(x1[i], x2[j], **self.kernel_params)

        return K


class GPRegression:
    def __init__(
        self,
        x_train: np.ndarray,
        gaussian_process: GP,
        noise: float = 1.0,
    ):

        self.x_train = x_train
        self.gp = gaussian_process
        self.noise = noise

        self.K_xx = None
        self.K_xX = None
        self.K_XX = None

        self.mu_pred = None
        self.cov_pred = None

    def fit(self, x_test: np.ndarray, y_test: np.ndarray):

        # make all kernel matrices
        self._make_kernel_matrices(x_test)

        # make posterior
        self.mu_pred = self.K_xX @ np.linalg.inv(self.K_XX) @ y_test
        self.cov_pred = self.K_xx - self.K_xX @ np.linalg.inv(self.K_XX) @ self.K_xX.T

    def sample_posterior(self) -> np.ndarray:
        assert self.mu_pred is not None and self.cov_pred is not None, "Fit model first"
        return np.random.multivariate_normal(self.mu_pred, self.cov_pred)

    def _make_kernel_matrices(self, x_test):
        self.K_XX = self.gp.make_gram_matrix(self.x_train, self.x_train)
        self.K_XX += np.eye(self.x_train.shape[0]) * self.noise

        self.K_xX = self.gp.make_gram_matrix(x_test, self.x_train)
        self.K_xx = self.gp.make_gram_matrix(x_test, x_test)


if __name__ == "__main__":
    from core.data.data_loader import load_data
    from core.models.kernels import covariance_kernel

    x, y = load_data()
    x = x[:, 0]

    kernel_params = {
        "theta": 8,
        "phi": 0.2,
        "psi": 0.001,
        "sigma": 5,
        "eta": 5,
        "tau": 1
    }

    gp = GP(mean=zero_mean, kernel=covariance_kernel, kernel_params=kernel_params)
    K = gp.make_gram_matrix(x, x)
    f_x = gp.sample(x, K)

    plt.figure()
    plt.plot(x, f_x, lw=2)
    plt.show()
