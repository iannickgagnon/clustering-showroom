from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SOMConfig:
    """Configuration for training a Self-Organizing Map (SOM).

    Attributes:
        n_rows (int): Number of rows in the SOM grid.
        n_cols (int): Number of columns in the SOM grid.
        input_vector_dim (int): Dimensionality of input vectors.
        initial_learning_rate (float, optional): Initial learning rate. Defaults to 0.5.
        initial_neighborhood_radius (Optional[float], optional): Initial neighborhood radius. Defaults to None.
        n_iter (int, optional): Number of training iterations. Defaults to 4000.
        rnd_seed (int, optional): Random seed for initialization and sampling. Defaults to 0.
    """

    n_rows: int
    n_cols: int
    input_vector_dim: int
    initial_learning_rate: float = 0.5
    initial_neighborhood_radius: Optional[float] = None
    n_iter: int = 4000
    rnd_seed: int = 0