"""
skmd.loop — SKMD outer-loop regime switching.

The outer loop operates in two phases:
  - Exploration (it < relax_start): high kernel amplitude, adaptive b
  - Relaxation  (it >= relax_start): decreasing amplitude and b
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def median_pairwise_distance(qs: list[np.ndarray]) -> float:
    """Median pairwise Euclidean distance between descriptor vectors (0.0 if < 2).

    numpy port of median_pairwise_distance in skmd-lammps (skmd_kernel.h);
    used once per run for the burn-in baseline b0.
    """
    if len(qs) < 2:
        return 0.0
    return float(np.median(pdist(np.asarray(qs))))


def get_regime_params(
    it: int,
    n_outer: int,
    relax_start: int,
    a_max: float,
    a_relax_mult: float,
    frozen_b0: float,
    b_relax_mult: float,
) -> tuple[str, float, str, float]:
    """Compute the kernel parameters for outer-loop iteration `it`.

    Args:
        it:            Current outer-loop iteration index (0-based).
        n_outer:       Total number of outer iterations.
        relax_start:   Iteration at which relaxation begins.
        a_max:         Kernel amplitude during exploration.
        a_relax_mult:  Final relaxation a as a fraction of a_max.
        frozen_b0:     Bandwidth b at the start of relaxation (typically the
                       last adaptive b seen during exploration).
        b_relax_mult:  Final relaxation b as a fraction of frozen_b0.

    Returns:
        (regime, a_curr, b_spec, relax_progress) where:
          regime        — 'exploration' or 'relaxation'
          a_curr        — kernel amplitude for this iteration
          b_spec        — 'adaptive' or a numeric string (float)
          relax_progress — float in [0, 1] (0 during exploration)
    """
    if it < relax_start:
        return "exploration", a_max, "adaptive", 0.0

    n_relax = max(1, n_outer - relax_start)
    progress = min(1.0, (it - relax_start + 1) / n_relax)

    a_final = a_max * a_relax_mult
    b_final = frozen_b0 * b_relax_mult

    a_curr = a_max + progress * (a_final - a_max)
    b_curr = frozen_b0 + progress * (b_final - frozen_b0)

    a_curr = max(a_curr, 1.0e-12)
    b_curr = max(b_curr, 1.0e-12)

    return "relaxation", a_curr, f"{b_curr}", progress


# Defensive lower clamp on adaptive b, as a multiple of the burn-in baseline b0.
B_MIN_MULT = 0.2


def normalize_b_mode(b_mode: str) -> str:
    """Map the legacy 'adaptive' spelling to its synonym 'adaptive-median'."""
    return "adaptive-median" if b_mode == "adaptive" else b_mode


def select_b(
    b_mode: str,
    b_user: float,
    b0: float,
    b_max_mult: float,
    b_min_mult: float = B_MIN_MULT,
) -> tuple[str, float | None, float | None]:
    """Choose the exploration-phase kernel bandwidth spec for fix skmd.

    Args:
        b_mode:     'fixed_user' (b_user), 'fixed_burnin' (b0), or
                    'adaptive'/'adaptive-median' (plugin recomputes b each
                    iteration, clamped to [b0*b_min_mult, b0*b_max_mult]).
        b_user:     User-supplied bandwidth (--b). Only used for fixed_user.
        b0:         Burn-in baseline bandwidth (median pairwise distance).
        b_max_mult: Upper clamp multiplier on b0. Only used for adaptive.
        b_min_mult: Lower clamp multiplier on b0. Only used for adaptive.

    Returns:
        (b_spec, b_max, b_min) — b_max/b_min are None unless adaptive.
    """
    b_mode = normalize_b_mode(b_mode)
    if b_mode == "fixed_user":
        return f"{b_user}", None, None
    elif b_mode == "fixed_burnin":
        return f"{b0}", None, None
    else:  # adaptive-median
        return "adaptive", b0 * b_max_mult, b0 * b_min_mult

