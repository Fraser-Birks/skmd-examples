"""Shared figure style for the AL learning-curve plotters, matching the paper's Muller-Brown figure."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt

FS = 20
PANEL_FIGSIZE = (15.5, 3.5)  # two panels side by side
SINGLE_FIGSIZE = (7.75, 3.5)  # one panel, same height and per-panel width
DPI = 600

# Arial when installed (as in the paper figure); falls back silently
# instead of spamming findfont warnings on machines without it (e.g. HPC).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["text.usetex"] = False

# label -> colour, shared across systems; UHAL takes UDD's colour from the Muller-Brown figure.
ARM_COLORS = {
    "a-SKMD": "blueviolet",
    "SKMD": "tab:green",
    "UHAL": "tab:orange",
    "uHAL": "tab:orange",
    "UHAL-offline": "tab:brown",
    "baseline (overdamped 50K)": "tab:gray",
    "Langevin 50K": "tab:gray",
    "Langevin 300K": "tab:blue",
    "Langevin 500K": "tab:cyan",
    "Langevin 700K": "navy",
}
_FALLBACK_COLORS = ["tab:red", "tab:pink", "tab:olive", "tab:purple", "gold", "teal"]


def arm_color(label: str, i: int) -> str:
    """Fixed colour for a known arm label (ignoring a trailing dynamics tag such
    as " (U)" or " (O)"); otherwise the i-th fallback."""
    base = re.sub(r"\s*\((U|O)\)$", "", label)
    return ARM_COLORS.get(base, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def new_figure(n_panels: int):
    """(fig, [ax, ...]) with the shared layout."""
    figsize = PANEL_FIGSIZE if n_panels == 2 else SINGLE_FIGSIZE
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=DPI, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.9, bottom=0.1, left=0.1, right=0.9)
    return fig, list(axes[0])


def plot_curve(ax, its, med, lo=None, hi=None, *, color, label, dashed=False) -> None:
    """Median line with markers, plus the 25th-75th percentile band if
    lo/hi are given (omitted for single-run curves)."""
    ax.plot(its, med, "o--" if dashed else "o-", markersize=3.5, linewidth=2.5,
            color=color, label=label)
    if lo is not None and hi is not None:
        ax.fill_between(its, lo, hi, color=color, edgecolor="none", alpha=0.15)


def style_axis(ax, title: str, ylabel: str, xlabel: str = "training iteration") -> None:
    ax.set_title(title, fontsize=FS + 2)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(axis="both", pad=2, labelsize=FS - 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)


def add_legend(fig, ax) -> None:
    """One legend for the whole figure, centred below the panels, built
    from `ax`'s handles (every panel carries the same curves)."""
    handles, labels = ax.get_legend_handles_labels()
    ncol = 4 if fig.get_figwidth() > SINGLE_FIGSIZE[0] else 2
    # matplotlib fills legend columns top-to-bottom; reorder so entries
    # read left-to-right across rows in plot order instead.
    nrows = -(-len(handles) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrows) if r * ncol + c < len(handles)]
    handles, labels = [handles[k] for k in order], [labels[k] for k in order]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=ncol, fontsize=FS - 2, frameon=False)


def save(fig, out_path: str) -> None:
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
