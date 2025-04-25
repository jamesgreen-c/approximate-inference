from core.data.data_loader import load_data
from core.models.gaussian_process import GPRegression, GP, zero_mean
from core.models.kernels import covariance_kernel
import numpy as np
import matplotlib.pyplot as plt


def bayesian_regression(X, Y, mu, sigma):

    # calculate posterior
    sigma_post = np.linalg.inv(np.linalg.inv(sigma) + X.T @ X)
    mu_post = sigma_post @ ((np.linalg.inv(sigma) @ mu) + (X.T @ Y).reshape(-1, 1))
    return mu_post, sigma_post


def plot_bayesian_regression(X, Y, mu_post, sigma_post):

    n = X.shape[0]

    # plot Data and learned MAP
    plt.figure(figsize=(9, 6))
    plt.plot(X[:, 0], Y)
    plt.plot(X[:, 0], np.repeat(360, n), label="Prior f(t)")
    plt.plot(X[:, 0], (X[:, 0] * mu_post[0, 0]) + np.repeat(mu_post[1, 0], n), label="Posterior f(t)")
    plt.legend()
    plt.xlabel("Decimal year", font={"size": 13})
    plt.ylabel("Parts per Million", font={"size": 13})
    plt.show()


def main():

    # default params
    mu = np.array([[0], [360]])
    sigma = np.array([[10 ** 2, 0], [0, 100 ** 2]])
    kernel_params = {
        "theta": 8,
        "phi": 0.2,
        "psi": 0.001,
        "sigma": 5,
        "eta": 5,
        "tau": 1
    }

    # regression
    X, Y = load_data()
    mu_post, sigma_post = bayesian_regression(X, Y, mu, sigma)

    # calculate residuals
    a = mu_post[0, 0]
    b = mu_post[1, 0]
    residuals = Y - (a * X[:, 0] + b)

    # GP regression on residuals
    x_obs = X[:, 0]
    x_new = np.arange(2007.75, 2021, 1 / 12)
    gp = GP(mean=zero_mean, kernel=covariance_kernel, kernel_params=kernel_params)
    gpr = GPRegression(x_train=x_obs, gaussian_process=gp)
    gpr.fit(x_new, residuals)

    # extrapolate using GP regression
    residuals_pred = gpr.sample_posterior()
    y_pred = (a * x_new) + b + residuals_pred
    std_deviations = np.sqrt(np.diag(gpr.cov_pred))

    # plot the extrapolation
    plt.figure(figsize=(11, 5))
    plt.plot(x_obs, Y, label="Observed")
    plt.plot(x_new, y_pred, c="red", label="Extrapolation")
    plt.plot(x_new, (a * x_new) + b, c="green", label="Predictive Mean")  # plot means
    plt.fill_between(x_new, y_pred - std_deviations, y_pred + std_deviations, color="orange", alpha=0.4,
                     label="1 STD")  # plot std deviation error
    plt.xlabel("Decimal Year")
    plt.ylabel("Parts per Million")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()






