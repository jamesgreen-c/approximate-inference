from core.data.data_loader import generate_data
from core.models.loopy_bp import LearnBinFactors
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    # number of data points - you can increase this if you want to learn better features (but it will take longer).
    N = 800
    D = 16  # dimensionality of the data
    Y = generate_data(N, D)

    nfeat = 8
    n_repeats = 5

    mu, sigma, pie, free_energies = None, None, None, None
    max_free_energy = -9999999999999999999999999999

    # repeat at different inits and choose best performing model
    for i in tqdm(range(n_repeats), desc="Running Loopy Interation"):

        _mu, _sigma, _pie, _free_energies = LearnBinFactors(Y, nfeat, 50)

        # take values if free energy is better
        if _free_energies[-1] > max_free_energy:
            mu, sigma, pie, free_energies = _mu, _sigma, _pie, _free_energies
            max_free_energy = free_energies[-1]

        # print(f"Free energy on interation {i}:", free_energies[-1])

    # plot learned means
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    fig.tight_layout()
    for i in range(nfeat):
        col = i % 4
        row = math.floor(i / 4)
        ax[row, col].imshow(mu[:, i].reshape((4, 4)), interpolation="None", cmap="gray")
        ax[row, col].axis("off")
    fig.patch.set_facecolor((0.82, 0.92, 0.82, 0.7))
    fig.show()

    # plot free energies
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(free_energies) + 1), free_energies)
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Free Energy", fontsize=18)
    plt.title("Best Free energy")
    plt.show()


if __name__ == "__main__":
    main()
