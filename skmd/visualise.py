"""
skmd.visualise — Generic visualisation utilities.

    fill_nan_by_interp  — interpolate NaN values in a 2-D grid
    pca_scatter         — scatter plot of PCA-projected descriptors
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fill_nan_by_interp(PHI, PSI, E, method="linear"):
    """Fill NaN entries in E by interpolating from finite neighbours.

    Args:
        PHI, PSI: 2-D coordinate grids (same shape as E).
        E:        2-D energy grid, may contain NaNs.
        method:   griddata interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        E_filled: copy of E with NaNs replaced.
    """
    mask_valid = np.isfinite(E)
    n_valid = int(np.sum(mask_valid))

    min_points = {"linear": 4, "nearest": 1, "cubic": 6}
    if n_valid < min_points.get(method, 4):
        method = "nearest"
    if n_valid < 1:
        return E.copy()

    points = np.column_stack([PHI[mask_valid], PSI[mask_valid]])
    values = E[mask_valid]
    E_filled = E.copy()
    mask_missing = ~mask_valid

    E_filled[mask_missing] = griddata(
        points, values,
        (PHI[mask_missing], PSI[mask_missing]),
        method=method,
    )

    still_missing = ~np.isfinite(E_filled)
    if np.any(still_missing):
        E_filled[still_missing] = griddata(
            points, values,
            (PHI[still_missing], PSI[still_missing]),
            method="nearest",
        )

    return E_filled


def pca_scatter(
    trajectories: list,
    descriptor_key: str = "mace_descriptor",
    n_components: int = 2,
    out_png: str | None = None,
    labels: list | None = None,
    s: float = 6,
    alpha: float = 0.5,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """PCA scatter of global descriptors across trajectory frames.

    Args:
        trajectories:   List of trajectory lists, each a list of ASE Atoms.
                        Each Atoms must have atoms.info[descriptor_key].
        descriptor_key: Key in atoms.info containing a 1-D descriptor array.
        n_components:   Number of PCA components, only used when fitting a
                        new PCA (i.e. when `pca` is None).
        out_png:        If given, save figure to this path.
        labels:         Optional list of labels per trajectory (for legend).
        s:              Scatter point size.
        alpha:          Scatter point opacity.
        scaler:         Pre-fitted StandardScaler to reuse, so plots share axes.
        pca:            Pre-fitted PCA to reuse, paired with `scaler`.

    Returns:
        (X_pca, pca, scaler) — projected coords and the PCA/scaler used.
    """
    all_desc = []
    traj_boundaries = []
    for traj in trajectories:
        start = len(all_desc)
        for atoms in traj:
            desc = np.asarray(atoms.info[descriptor_key], dtype=float).reshape(-1)
            all_desc.append(desc)
        traj_boundaries.append((start, len(all_desc)))

    X = np.stack(all_desc, axis=0)

    if scaler is None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
    else:
        X_std = scaler.transform(X)

    if pca is None:
        n_comp = min(n_components, X_std.shape[0] - 1, X_std.shape[1])
        if n_comp < 2:
            # Need >= 3 frames for 2 components (common in early iterations).
            print(f"pca_scatter: skipping — {X_std.shape[0]} total frame(s), "
                  f"need at least 3 to plot 2 PCA components.")
            return None, None, scaler
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_std)
    else:
        X_pca = pca.transform(X_std)

    fig, ax = plt.subplots()
    for t_idx, (start, end) in enumerate(traj_boundaries):
        label = labels[t_idx] if labels is not None else f"traj {t_idx}"
        ax.scatter(X_pca[start:end, 0], X_pca[start:end, 1], s=s, alpha=alpha, label=label)

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA of global descriptors")
    if labels is not None:
        ax.legend(markerscale=2)
    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return X_pca, pca, scaler
