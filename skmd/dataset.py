"""
skmd.dataset — .skdat file I/O for SKMD descriptor/force/jacobian datasets.

File format: three plain-text files per dataset base path.
  <base>_qs.skdat   — descriptor vectors, shape (npts, ncoeff)
  <base>_fs.skdat   — force vectors, shape (npts, N, 3)
  <base>_jacs.skdat — jacobian matrices, shape (npts, 3N, ncoeff)
"""

from __future__ import annotations
import numpy as np


def write_skdat(base: str, qs: list, fs: list, jacs: list) -> None:
    """Write a SKMD dataset to three .skdat text files.

    Args:
        base: File path prefix (suffixes _qs, _fs, _jacs are appended).
        qs:   List of npts arrays of shape (ncoeff,).
        fs:   List of npts arrays of shape (N, 3).
        jacs: List of npts arrays of shape (3N, ncoeff).
    """
    npts = len(qs)
    if npts == 0:
        open(f"{base}_qs.skdat",   "w").write("0 0\n")
        open(f"{base}_fs.skdat",   "w").write("0 0\n")
        open(f"{base}_jacs.skdat", "w").write("0 0 0\n")
        return

    ncoeff = int(len(qs[0]))
    N      = int(fs[0].shape[0])

    with open(f"{base}_qs.skdat", "w") as fp:
        fp.write(f"{npts} {ncoeff}\n")
        for q in qs:
            fp.write(" ".join(map(str, np.asarray(q).tolist())) + "\n")

    with open(f"{base}_fs.skdat", "w") as fp:
        fp.write(f"{npts} {N}\n")
        for f in fs:
            fp.write(" ".join(map(str, np.asarray(f).reshape(-1).tolist())) + "\n")

    with open(f"{base}_jacs.skdat", "w") as fp:
        fp.write(f"{npts} {3 * N} {ncoeff}\n")
        for j in jacs:
            fp.write(" ".join(map(str, np.asarray(j).reshape(-1).tolist())) + "\n")


def read_skdat(base: str) -> tuple[list, list, list]:
    """Read a SKMD dataset from three .skdat text files.

    Args:
        base: File path prefix used in write_skdat.

    Returns:
        (qs, fs, jacs) — lists of numpy arrays matching the shapes written.
    """
    with open(f"{base}_qs.skdat") as fp:
        header = fp.readline().split()
        npts, ncoeff = int(header[0]), int(header[1])
        if npts == 0:
            return [], [], []
        qs = [np.fromstring(fp.readline(), sep=" ") for _ in range(npts)]

    with open(f"{base}_fs.skdat") as fp:
        header = fp.readline().split()
        N = int(header[1])
        fs = [np.fromstring(fp.readline(), sep=" ").reshape(N, 3) for _ in range(npts)]

    with open(f"{base}_jacs.skdat") as fp:
        fp.readline()  # header: npts 3N ncoeff
        jacs = [np.fromstring(fp.readline(), sep=" ").reshape(3 * N, ncoeff) for _ in range(npts)]

    return qs, fs, jacs
