import numpy as np
import matplotlib.pyplot as plt

from self_organizing_maps.classes.self_organizing_map import SelfOrganizingMap


def make_toy_blobs(seed: int = 1) -> np.ndarray:
    """Creates a simple 2D toy dataset (three Gaussian blobs).

    Args:
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        np.ndarray: Array of shape (900, 2) containing three clusters.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal([0.0, 0.0], 0.35, size=(300, 2))
    x2 = rng.normal([2.0, 0.0], 0.35, size=(300, 2))
    x3 = rng.normal([1.0, 1.6], 0.35, size=(300, 2))
    return np.vstack([x1, x2, x3]).astype(float)


def plot_codebook_over_data(X: np.ndarray, som: SelfOrganizingMap) -> None:
    """Plots training data and learned SOM codebook vectors.

    Args:
        X (np.ndarray): Data of shape (N, 2).
        som (SelfOrganizingMap): A trained SOM with dim=2.
    """
    W = som.weights.reshape(-1, 2)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.25)
    plt.scatter(W[:, 0], W[:, 1], s=35)
    plt.title("SOM codebook vectors over data")
    plt.show()


def plot_occupancy_map(occ: np.ndarray) -> None:
    """Plots the SOM occupancy map.

    Args:
        occ (np.ndarray): Occupancy matrix of shape (m, n).
    """
    plt.figure()
    plt.imshow(occ.T, origin="lower")
    plt.title("SOM occupancy (how many points per neuron)")
    plt.colorbar()
    plt.show()