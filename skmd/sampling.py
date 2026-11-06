"""Diverse structure selection by greedy farthest point sampling (FPS)."""
import numpy as np


def get_descriptor_matrix(atoms_list, key="mace_descriptor", normalise=False):
    """Build an (n_structures, d) array from atoms.info[key]."""
    X = np.stack([np.asarray(atoms.info[key], dtype=float) for atoms in atoms_list], axis=0)

    if X.ndim != 2:
        raise ValueError(f"Expected descriptor matrix of shape (n, d), got {X.shape}")

    if normalise:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        X = X / norms

    return X


def farthest_point_sampling(X, n_select, initial_features=None, initial="medoid", random_state=None):
    """Greedy farthest-point sampling: indices of n_select new rows of X, in selection order.

    initial_features (previously selected points) seed the min-distance field;
    otherwise `initial` ('random', 'medoid' or an index) picks the first point.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if not (1 <= n_select <= n):
        raise ValueError(f"n_select must satisfy 1 <= n_select <= {n}")

    rng = np.random.default_rng(random_state)
    selected = []

    if initial_features is not None:
        initial_features = np.asarray(initial_features, dtype=float)
        if initial_features.ndim != 2:
            raise ValueError("initial_features must have shape (n_initial, n_features)")
        if initial_features.shape[1] != X.shape[1]:
            raise ValueError(
                f"initial_features has feature dimension {initial_features.shape[1]}, "
                f"but X has feature dimension {X.shape[1]}"
            )

        # Initialise distances to the previously selected set
        min_d2 = np.full(n, np.inf)
        for x0 in initial_features:
            d2 = np.sum((X - x0) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

    else:
        if initial == "random":
            first = int(rng.integers(n))
        elif initial == "medoid":
            centroid = X.mean(axis=0)
            first = int(np.argmin(np.sum((X - centroid) ** 2, axis=1)))
        elif isinstance(initial, int):
            if not (0 <= initial < n):
                raise ValueError(f"initial index must be in [0, {n})")
            first = initial
        else:
            raise ValueError("initial must be 'random', 'medoid', or an integer index")

        selected = [first]
        min_d2 = np.sum((X - X[first]) ** 2, axis=1)
        min_d2[first] = -np.inf

    while len(selected) < n_select:
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)

        d2_new = np.sum((X - X[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[selected] = -np.inf

    return selected


def sample_diverse_structures(atoms_list, n_select, key="mace_descriptor", normalise=False,
                              previous_atoms_list=None, initial="medoid", random_state=None):
    """Select n_select diverse structures by FPS; previous_atoms_list seeds the distance field.

    Returns (selected_idx, selected_atoms).
    """
    X = get_descriptor_matrix(atoms_list, key=key, normalise=normalise)
    X_previous = None
    if previous_atoms_list is not None:
        X_previous = get_descriptor_matrix(previous_atoms_list, key=key, normalise=normalise)
    selected_idx = farthest_point_sampling(X, n_select=n_select, initial_features=X_previous,
                                           initial=initial, random_state=random_state)
    return selected_idx, [atoms_list[i] for i in selected_idx]
