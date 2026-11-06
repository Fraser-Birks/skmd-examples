import numpy as np
import tempfile
import os
from skmd.dataset import write_skdat, read_skdat


def test_round_trip_basic():
    """write_skdat then read_skdat returns identical arrays."""
    rng = np.random.default_rng(0)
    npts, ncoeff, N = 5, 12, 22

    qs   = [rng.random(ncoeff)          for _ in range(npts)]
    fs   = [rng.random((N, 3))          for _ in range(npts)]
    jacs = [rng.random((3 * N, ncoeff)) for _ in range(npts)]

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "dataset")
        write_skdat(base, qs, fs, jacs)
        qs_r, fs_r, jacs_r = read_skdat(base)

    assert len(qs_r) == npts
    assert len(fs_r) == npts
    assert len(jacs_r) == npts

    for i in range(npts):
        np.testing.assert_allclose(qs_r[i],   qs[i],   rtol=1e-12, err_msg=f"q mismatch at {i}")
        np.testing.assert_allclose(fs_r[i],   fs[i],   rtol=1e-12, err_msg=f"f mismatch at {i}")
        np.testing.assert_allclose(jacs_r[i], jacs[i], rtol=1e-12, err_msg=f"jac mismatch at {i}")


def test_empty_dataset():
    """write_skdat with zero points produces files readable by read_skdat."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "empty")
        write_skdat(base, [], [], [])
        qs_r, fs_r, jacs_r = read_skdat(base)

    assert qs_r == []
    assert fs_r == []
    assert jacs_r == []
