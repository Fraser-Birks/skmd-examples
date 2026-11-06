"""Checks for the Müller–Brown experiments in systems/muller_brown (needs torch)."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "systems" / "muller_brown"))

from utils.kernels import RBFKernel  # noqa: E402
from utils.potentials import MLPPotential, MullerBrown  # noqa: E402
from utils.simulators import SKMD  # noqa: E402


def _autograd_forces(energy_fn, X):
    X = X.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy_fn(X).sum(), X)
    return -grad


def test_muller_brown_forces_are_negative_energy_gradient():
    torch.manual_seed(0)
    X = torch.rand(20, 2) * torch.tensor([3.0, 2.5]) + torch.tensor([-1.5, -0.3])
    mb = MullerBrown()
    torch.testing.assert_close(mb.forces(X), _autograd_forces(mb.potential_energy, X), rtol=1e-4, atol=1e-4)


def test_muller_brown_minimum():
    # Lowest minimum of the Müller–Brown potential, (-0.558, 1.442), V = 0.05 * -146.70
    mb = MullerBrown(dtype=torch.float64)
    x = torch.tensor([-0.558224, 1.441726], dtype=torch.float64)
    assert mb.potential_energy(x).item() == pytest.approx(0.05 * -146.699517, rel=1e-6)
    assert torch.linalg.norm(mb.forces(x)).item() < 1e-3


def test_mlp_forces_match_energy_gradient():
    torch.manual_seed(0)
    model = MLPPotential(hidden_dim=20, n_hidden_layers=1, activation=torch.nn.Tanh).double()
    X = torch.randn(10, 2, dtype=torch.float64)
    # potential_energy detaches its input, so compare against central finite differences
    h, fd = 1e-6, torch.empty_like(X)
    for d in range(2):
        dX = torch.zeros_like(X)
        dX[:, d] = h
        fd[:, d] = -(model.potential_energy(X + dX) - model.potential_energy(X - dX)).squeeze(1).detach() / (2 * h)
    torch.testing.assert_close(model.forces(X).detach(), fd, rtol=1e-6, atol=1e-6)


def test_rbf_kernel_gradient():
    torch.manual_seed(0)
    kernel = RBFKernel(alpha=2.0, ls=0.41)
    X, x = torch.randn(8, 2), torch.randn(2)
    Xg = X.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(kernel.compute_kernel_batch(Xg, x).sum(), Xg)
    torch.testing.assert_close(kernel.compute_grad_kernel_batch(X, x), grad)


def test_skmd_bias_vanishes_with_zero_kernel_amplitude():
    """With a = 0 the SKMD step is drift-only overdamped Langevin under A * beta * F."""
    torch.manual_seed(0)
    mb = MullerBrown()
    sys0 = torch.randn(5, 2)
    sim = SKMD(kernel=RBFKernel(alpha=0.0, ls=0.41), set_kernel_band=lambda *a: None, sys=sys0,
               sys_fix=sys0, A=1.0, beta=1.0, dt=1e-3, T=3, device=None, seed=0)
    sim.diff_coeff = torch.tensor(0.0)
    x = torch.tensor([-0.5, 1.4])
    sim.logger["traj"][0] = x
    sim.sim_step(0, mb)
    torch.testing.assert_close(sim.logger["traj"][1], x + 1e-3 * mb.forces(x))
