from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from self_organizing_maps.classes.som_config import SOMConfig

ArrayF = np.ndarray



class SelfOrganizingMap:
    """A minimal 2D Kohonen Self-Organizing Map (SOM) implementation.

    This SOM uses:
    - Euclidean distance for BMU selection.
    - A Gaussian neighborhood function over the 2D neuron grid.
    - Exponential decay for learning rate and neighborhood radius.

    The implementation is intentionally minimal but fully working.
    """

    def __init__(self, config: SOMConfig) -> None:
        """Initializes a SOM with random weights and a 2D neuron grid.

        Args:
            config: Hyperparameters and training settings.
        """
        self._cfg = config
        self._rng = np.random.default_rng(config.rnd_seed)

        self._sigma0 = config.initial_neighborhood_radius if config.initial_neighborhood_radius is not None else max(config.n_rows, config.n_cols) / 2.0
        self._lr0 = config.initial_learning_rate

        # Weight matrix of shape (m, n, dim).
        self._weights: ArrayF = self._rng.normal(0.0, 1.0, size=(config.n_rows, config.n_cols, config.input_vector_dim)).astype(float)

        # Precompute neuron coordinates (m, n, 2).
        xs, ys = np.meshgrid(np.arange(config.n_rows), np.arange(config.n_cols), indexing="ij")
        self._coords: ArrayF = np.stack([xs, ys], axis=-1).astype(float)

    @property
    def weights(self) -> ArrayF:
        """Returns the SOM weight/codebook vectors.

        Returns:
            The weight matrix with shape (m, n, dim).
        """
        return self._weights

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the SOM grid shape.

        Returns:
            (m, n) grid shape.
        """
        return (self._cfg.n_rows, self._cfg.n_cols)

    def fit(self, X: ArrayF) -> SelfOrganizingMap:
        """Trains the SOM on input data.

        Args:
            X: Training data of shape (num_samples, dim).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If X does not have shape (N, dim).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._cfg.input_vector_dim:
            raise ValueError(f"Expected X shape (N, {self._cfg.input_vector_dim}), got {X.shape}")

        n_samples = X.shape[0]

        for t in range(self._cfg.n_iter):
            # Exponential decay schedules.
            lr = self._lr0 * np.exp(-t / self._cfg.n_iter)
            sigma = self._sigma0 * np.exp(-t / self._cfg.n_iter)

            x = X[self._rng.integers(0, n_samples)]
            bmu_i, bmu_j = self._find_bmu(x)

            bmu_coord = np.array([bmu_i, bmu_j], dtype=float)

            # Gaussian neighborhood in grid space.
            dist2 = np.sum((self._coords - bmu_coord) ** 2, axis=2)  # (m, n)
            h = np.exp(-dist2 / (2.0 * (sigma ** 2) + 1e-12))        # (m, n)

            # Update weights: W_ij += lr * h_ij * (x - W_ij).
            self._weights += lr * h[..., None] * (x - self._weights)

        return self

    def map_points(self, X: ArrayF) -> ArrayF:
        """Maps each input vector to its Best Matching Unit (BMU) grid coordinate.

        Args:
            X: Input data of shape (num_samples, dim).

        Returns:
            Array of BMU indices with shape (num_samples, 2) where each row is (i, j).

        Raises:
            ValueError: If X does not have shape (N, dim).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._cfg.input_vector_dim:
            raise ValueError(f"Expected X shape (N, {self._cfg.input_vector_dim}), got {X.shape}")

        bmus = np.zeros((X.shape[0], 2), dtype=int)
        for idx, x in enumerate(X):
            i, j = self._find_bmu(x)
            bmus[idx] = (i, j)
        return bmus

    def occupancy(self, X: ArrayF) -> ArrayF:
        """Computes the occupancy map (wins per neuron) for a dataset.

        Args:
            X: Input data of shape (num_samples, dim).

        Returns:
            Occupancy matrix of shape (m, n) where each cell counts mapped points.
        """
        bmus = self.map_points(X)
        occ = np.zeros((self._cfg.n_rows, self._cfg.n_cols), dtype=int)
        for i, j in bmus:
            occ[i, j] += 1
        return occ

    def _find_bmu(self, x: ArrayF) -> Tuple[int, int]:
        """Finds the Best Matching Unit (BMU) for a single vector.

        Args:
            x: Input vector of shape (dim,).

        Returns:
            Tuple (i, j) index of the BMU in the (m, n) grid.
        """
        # Squared Euclidean distances to all neurons.
        d2 = np.sum((self._weights - x) ** 2, axis=2)  # (m, n)
        flat_index = int(np.argmin(d2))
        return np.unravel_index(flat_index, d2.shape)


def make_toy_blobs(seed: int = 1) -> ArrayF:
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


def plot_codebook_over_data(X: ArrayF, som: SelfOrganizingMap) -> None:
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


def plot_occupancy_map(occ: ArrayF) -> None:
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
