import numpy as np
import matplotlib.pyplot as plt

from self_organizing_maps.classes.som_config import SOMConfig
from self_organizing_maps.classes.self_organizing_map import SelfOrganizingMap


def make_toy_blobs(seed: int = 1) -> np.ndarray:
    """Creates a simple 2D toy dataset (three Gaussian blobs).

    Args:
        seed: Random seed.

    Returns:
        Array of shape (900, 2) containing three clusters.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal([0.0, 0.0], 0.35, size=(300, 2))
    x2 = rng.normal([2.0, 0.0], 0.35, size=(300, 2))
    x3 = rng.normal([1.0, 1.6], 0.35, size=(300, 2))
    return np.vstack([x1, x2, x3]).astype(float)


def plot_codebook_over_data(X: np.ndarray, som: SelfOrganizingMap) -> None:
    """Plots training data and learned SOM codebook vectors.

    Args:
        X: Data of shape (N, 2).
        som: A trained SOM with dim=2.
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
        occ: Occupancy matrix of shape (m, n).
    """
    plt.figure()
    plt.imshow(occ.T, origin="lower")
    plt.title("SOM occupancy (how many points per neuron)")
    plt.colorbar()
    plt.show()


def main() -> None:
    """Runner example: train a SOM on toy data and visualize results."""
    X = make_toy_blobs(seed=2)

    cfg = SOMConfig(
        n_rows=15,
        n_cols=15,
        input_vector_dim=2,
        initial_learning_rate=0.5,
        initial_neighborhood_radius=6.0,
        n_iter=4000,
        rnd_seed=0,
    )

    som = SelfOrganizingMap(cfg).fit(X)

    plot_codebook_over_data(X, som)
    occ = som.occupancy(X)
    plot_occupancy_map(occ)


if __name__ == "__main__":
    main()
