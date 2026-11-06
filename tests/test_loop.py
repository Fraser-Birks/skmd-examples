import numpy as np

from skmd.loop import get_regime_params, median_pairwise_distance


def test_exploration_regime():
    regime, a, b_spec, progress = get_regime_params(
        it=0, n_outer=200, relax_start=100,
        a_max=1.0, a_relax_mult=0.1,
        frozen_b0=0.5, b_relax_mult=0.1,
    )
    assert regime == "exploration"
    assert a == 1.0
    assert b_spec == "adaptive"
    assert progress == 0.0


def test_exploration_just_before_boundary():
    regime, a, b_spec, progress = get_regime_params(
        it=99, n_outer=200, relax_start=100,
        a_max=1.0, a_relax_mult=0.1,
        frozen_b0=0.5, b_relax_mult=0.1,
    )
    assert regime == "exploration"
    assert b_spec == "adaptive"


def test_relaxation_at_boundary():
    regime, a, b_spec, progress = get_regime_params(
        it=100, n_outer=200, relax_start=100,
        a_max=1.0, a_relax_mult=0.1,
        frozen_b0=0.5, b_relax_mult=0.1,
    )
    assert regime == "relaxation"
    # b_spec should be a numeric string, not 'adaptive'
    assert b_spec != "adaptive"
    float(b_spec)  # must be parseable as a float


def test_relaxation_a_smaller_than_exploration():
    _, a_exp, _, _ = get_regime_params(
        it=0, n_outer=200, relax_start=100,
        a_max=1.0, a_relax_mult=0.1,
        frozen_b0=0.5, b_relax_mult=0.1,
    )
    _, a_relax, _, _ = get_regime_params(
        it=100, n_outer=200, relax_start=100,
        a_max=1.0, a_relax_mult=0.1,
        frozen_b0=0.5, b_relax_mult=0.1,
    )
    assert a_relax < a_exp


def test_a_always_positive():
    for it in [0, 50, 100, 150, 199]:
        _, a, _, _ = get_regime_params(
            it=it, n_outer=200, relax_start=100,
            a_max=1.0, a_relax_mult=0.1,
            frozen_b0=0.5, b_relax_mult=0.1,
        )
        assert a > 0.0


def test_median_pairwise_distance_fewer_than_two_points():
    assert median_pairwise_distance([]) == 0.0
    assert median_pairwise_distance([np.array([1.0, 2.0])]) == 0.0


def test_median_pairwise_distance_two_points():
    # Only one pair, distance 5 (3-4-5 triangle).
    qs = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
    assert median_pairwise_distance(qs) == 5.0


def test_median_pairwise_distance_known_set():
    # Pairwise distances 1, 2, 3 -> median 2.
    qs = [np.array([0.0]), np.array([1.0]), np.array([3.0])]
    assert median_pairwise_distance(qs) == 2.0


def test_median_pairwise_distance_even_pair_count_averages():
    # Pairwise distances [1, 1, 2, 2, 3, 4] -> median 2.0.
    qs = [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([4.0])]
    assert median_pairwise_distance(qs) == 2.0
