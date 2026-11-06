import numpy as np
import pytest
from ase import Atoms
from skmd.sampling import farthest_point_sampling, get_descriptor_matrix, sample_diverse_structures


def _make_atoms_with_descriptors(n_structures, d=8, seed=42):
    """Create a list of ASE Atoms objects with synthetic descriptor data."""
    rng = np.random.default_rng(seed)
    atoms_list = []
    for _ in range(n_structures):
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        a.info["mace_descriptor"] = rng.random(d).astype(float)
        atoms_list.append(a)
    return atoms_list


def test_get_descriptor_matrix_shape():
    atoms_list = _make_atoms_with_descriptors(10, d=8)
    X = get_descriptor_matrix(atoms_list)
    assert X.shape == (10, 8)


def test_farthest_point_sampling_returns_valid_indices():
    rng = np.random.default_rng(0)
    X = rng.random((20, 5))
    idx = farthest_point_sampling(X, n_select=5)
    assert len(idx) == 5
    assert all(isinstance(i, (int, np.integer)) for i in idx)
    assert all(0 <= i < 20 for i in idx)
    assert len(set(int(i) for i in idx)) == 5  # no duplicates


def test_sample_diverse_structures_output_shapes():
    atoms_list = _make_atoms_with_descriptors(30, d=12)
    idx, selected = sample_diverse_structures(atoms_list, n_select=5)
    assert len(idx) == 5
    assert len(selected) == 5
    assert all(0 <= i < 30 for i in idx)
    assert len(set(idx)) == 5


def test_previous_structures_seed_fps():
    # Points duplicating a previous selection must be picked last.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    idx = farthest_point_sampling(X, n_select=4, initial_features=np.array([[5.0, 5.0]]))
    assert idx[-1] == 3
    assert idx[0] == 0
