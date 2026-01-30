## SOM Initialization (`__init__`)

The `__init__` method sets up the Self-Organizing Map's internal data structures and initializes the neural network before training.

### Configuration Storage
- Stores the `SOMConfig` object for later access to hyperparameters during training.

### Random Number Generator
- Creates a seeded NumPy random number generator (`np.random.default_rng`) using `config.rnd_seed` to ensure reproducible initialization and training.

### Initial Neighborhood Radius (σ₀)
- Sets the initial neighborhood radius `_sigma0`:
  - If `config.initial_neighborhood_radius` is provided, uses that value.
  - Otherwise, defaults to `max(n_rows, n_cols) / 2.0` (half the larger grid dimension).
- This controls how far the neighborhood influence extends during early training iterations.

### Initial Learning Rate (α₀)
- Stores `config.initial_learning_rate` as `_lr0`.
- This controls how much neurons update their weights toward input vectors during training.

### Weight Matrix Initialization
- Creates a 3D weight matrix with shape `(n_rows, n_cols, input_vector_dim)`:
  - Each neuron `(i, j)` has a weight vector of length `input_vector_dim`.
  - Weights are initialized from a normal distribution with mean 0.0 and standard deviation 1.0.
  - These random weights serve as the initial codebook vectors before training.

### Neuron Coordinate Precomputation
- Precomputes 2D grid coordinates for all neurons:
  - Uses `np.meshgrid` to create coordinate arrays for rows and columns.
  - Stacks them into a 3D array `(n_rows, n_cols, 2)` where each neuron's position `[i, j]` stores its `[x, y]` grid coordinates.
  - This avoids recomputing distances during training when finding the Best Matching Unit (BMU) and calculating neighborhood functions.