from self_organizing_maps.classes.som_config import SOMConfig
from self_organizing_maps.classes.self_organizing_map import SelfOrganizingMap
from utils import make_toy_blobs, plot_codebook_over_data, plot_occupancy_map

if __name__ == "__main__":

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
