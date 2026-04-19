"""Shared utilities for scaled-vs-unscaled PINN experiments.

Problem (distributed Poisson OCP on Omega = (0,1)^2):
    min 0.5 * ||y - y_d||^2 + 0.5 * alpha * ||u - u_d||^2
    s.t. -Delta y = f + u in Omega,  y = 0 on bdry.

After eliminating u = u_d - p/alpha:

Unscaled system (1.4):
    r1 = -Delta y - (f + u_d) + (1/alpha) * p
    r2 = -Delta p -  y        +           y_d

Scaled system (1.5) [symmetric sqrt(alpha) rescaling]:
    r1_s = -alpha^{1/2} Delta y + p - alpha^{3/4} (f + u_d)
    r2_s = -alpha^{1/2} Delta p - y + alpha^{1/4}  y_d
    loss_s = mean((r1_s / alpha^{3/4})^2) + mean((r2_s / alpha^{1/4})^2)

Manufactured solution chosen consistent with the KKT system:
    y*(x)  = sin(pi x1) sin(pi x2)
    p*(x)  = alpha * sin(pi x1) sin(pi x2)
    u*(x)  = -p*/alpha = -sin(pi x1) sin(pi x2)
Then, requiring both residuals to vanish at (y*, p*):
    f + u_d = (2 pi^2 + 1) * y*    (so source_f below absorbs u_d = 0)
    y_d     = (1 - 2 pi^2 alpha) * y*
Boundary: y* = p* = 0 on bdry (hard BC ansatz used).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS > CPU. Override via env var PINN_DEVICE or
# the `prefer` argument.  CUDA is checked first so Windows+NVIDIA setups
# (the primary target for this branch) land on GPU automatically.
# ---------------------------------------------------------------------------
def pick_device(prefer: str | None = None) -> torch.device:
    import os
    if prefer is None:
        prefer = os.environ.get("PINN_DEVICE")
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_float64(device: torch.device) -> bool:
    """MPS (as of PyTorch 2.x) has partial float64 support. Force CPU for f64."""
    return device.type != "mps"


# ---------------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------------
@dataclass
class ManufacturedSolution:
    """Analytic ground truth for the Poisson OCP with given alpha."""

    alpha: float

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

    def exact_y(self, x: torch.Tensor) -> torch.Tensor:
        return self._phi(x)

    def exact_p(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self._phi(x)

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        return -self._phi(x)

    def exact_y_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """y_scaled = alpha^{1/4} * y_bar_star = alpha^{1/4} * phi."""
        return self.alpha ** 0.25 * self._phi(x)

    def exact_p_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """p_scaled = alpha^{-1/4} * p_bar_star = alpha^{3/4} * phi."""
        return self.alpha ** 0.75 * self._phi(x)

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        # f + u_d = (2 pi^2 + 1) phi, with u_d = 0.
        return (2.0 * math.pi ** 2 + 1.0) * self._phi(x)

    def prior_ud(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0:1])

    def target_yd(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - 2.0 * math.pi ** 2 * self.alpha) * self._phi(x)


# ---------------------------------------------------------------------------
# Networks: 4 hidden layers x 50 units, tanh, Glorot init.
# Two separate MLPs (one for y, one for p) matching the "dual-net" design.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, width: int = 50, depth: int = 4, out_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim] + [width] * depth + [out_dim]
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_networks(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
    width: int = 50,
    depth: int = 4,
) -> tuple[MLP, MLP]:
    if seed is not None:
        torch.manual_seed(seed)
    net_y = MLP(width=width, depth=depth).to(device=device, dtype=dtype)
    net_p = MLP(width=width, depth=depth).to(device=device, dtype=dtype)
    return net_y, net_p


def hard_bc(x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Multiply raw network output by x1(1-x1) x2(1-x2) to enforce zero BC."""
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return raw * (x1 * (1.0 - x1) * x2 * (1.0 - x2))


# ---------------------------------------------------------------------------
# Sampling. Latin-hypercube interior + boundary samples (reproducible).
# ---------------------------------------------------------------------------
def sample_interior(n: int, seed: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # SciPy changed this API from `seed` to `rng` across versions.
    lhs_ctor = cast(Any, qmc.LatinHypercube)
    try:
        sampler = lhs_ctor(d=2, rng=np.random.default_rng(seed))
    except TypeError:
        sampler = lhs_ctor(d=2, seed=seed)
    pts = sampler.random(n)
    return torch.as_tensor(pts, dtype=dtype, device=device)


def sample_boundary(n_per_side: int, seed: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n_per_side, 1))
    sides = [
        np.hstack([t, np.zeros_like(t)]),
        np.hstack([t, np.ones_like(t)]),
        np.hstack([np.zeros_like(t), t]),
        np.hstack([np.ones_like(t), t]),
    ]
    pts = np.vstack(sides)
    return torch.as_tensor(pts, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------------
def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        gi = grad_u[:, i:i + 1]
        u_ii = torch.autograd.grad(gi, x, grad_outputs=torch.ones_like(gi), create_graph=True)[0][:, i:i + 1]
        lap = lap + u_ii
    return lap


# ---------------------------------------------------------------------------
# Predictions with hard BC ansatz
# ---------------------------------------------------------------------------
def predict(net_y: nn.Module, net_p: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = hard_bc(x, net_y(x))
    p = hard_bc(x, net_p(x))
    return y, p


# ---------------------------------------------------------------------------
# Residuals — raw (pointwise) tensors, shape (N, 1). Sum-of-mean-squares is
# computed by the loss helpers below so that experiments can also access the
# residual vector directly (for Jacobian assembly, per-term diagnostics, etc).
# ---------------------------------------------------------------------------
Formulation = Literal["unscaled", "scaled_normalized", "scaled_raw"]


def residuals(
    net_y: nn.Module,
    net_p: nn.Module,
    x: torch.Tensor,
    mms: ManufacturedSolution,
    formulation: Formulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (r1, r2) as (N,1) tensors for the given formulation."""
    x = x.detach().clone().requires_grad_(True)
    y, p = predict(net_y, net_p, x)
    lap_y = laplacian(y, x)
    lap_p = laplacian(p, x)
    f = mms.source_f(x)
    ud = mms.prior_ud(x)
    yd = mms.target_yd(x)
    a = mms.alpha

    if formulation == "unscaled":
        r1 = -lap_y - (f + ud) + (1.0 / a) * p
        r2 = -lap_p - y + yd
    elif formulation == "scaled_normalized":
        # HISTORICAL: loss-level preconditioning that divides by a^{3/4} and
        # a^{1/4}. Kept as a reference; no experiment should use this.
        a12, a34, a14 = a ** 0.5, a ** 0.75, a ** 0.25
        r1 = (-a12 * lap_y + p - a34 * (f + ud)) / a34
        r2 = (-a12 * lap_p - y + a14 * yd) / a14
    elif formulation == "scaled_raw":
        # Raw residuals of system (1.5), WITHOUT the loss-level normalisation
        # by a34 / a14. Use this for spectral diagnostics (Exp 1: Jacobian
        # conditioning; Exp 2: gradient balance) so that the measured kappa(J)
        # matches the theoretical alpha-scaling prediction. Training losses
        # "scaled_normalized" keeps the division as a historical reference but
        # all current experiments use "scaled_raw" for both training and diagnostics.
        a12, a34, a14 = a ** 0.5, a ** 0.75, a ** 0.25
        r1 = -a12 * lap_y + p - a34 * (f + ud)
        r2 = -a12 * lap_p - y + a14 * yd
    else:  # pragma: no cover
        raise ValueError(formulation)
    return r1, r2


def loss_terms(
    net_y: nn.Module,
    net_p: nn.Module,
    x: torch.Tensor,
    mms: ManufacturedSolution,
    formulation: Formulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    r1, r2 = residuals(net_y, net_p, x, mms, formulation)
    return (r1 ** 2).mean(), (r2 ** 2).mean()


def total_loss(
    net_y: nn.Module,
    net_p: nn.Module,
    x: torch.Tensor,
    mms: ManufacturedSolution,
    formulation: Formulation,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    l1, l2 = loss_terms(net_y, net_p, x, mms, formulation)
    return w1 * l1 + w2 * l2


# ---------------------------------------------------------------------------
# Error metrics (relative L2 on a regular grid)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _grid_eval(net_y: nn.Module, net_p: nn.Module, mms: ManufacturedSolution, device: torch.device, n: int = 100):
    g = torch.linspace(0, 1, n, device=device, dtype=next(net_y.parameters()).dtype)
    x1, x2 = torch.meshgrid(g, g, indexing="ij")
    x = torch.stack([x1.flatten(), x2.flatten()], dim=-1)
    y, p = predict(net_y, net_p, x)
    return x, y, p


def relative_l2_errors(
    net_y: nn.Module,
    net_p: nn.Module,
    mms: ManufacturedSolution,
    device: torch.device,
    formulation: Formulation = "unscaled",
    n: int = 100,
) -> dict[str, float]:
    x, y_pred, p_pred = _grid_eval(net_y, net_p, mms, device, n)

    # When training with "scaled_raw", the networks learn the scaled variables
    #   y_net = alpha^{1/4} y_bar,   p_net = alpha^{3/4} p_bar / alpha = alpha^{-1/4} p_bar.
    # Convert back to physical (unscaled) variables before computing errors.
    if formulation == "scaled_raw":
        y_phys = y_pred / mms.alpha ** 0.25
        p_phys = p_pred * mms.alpha ** 0.25
    else:
        y_phys = y_pred
        p_phys = p_pred

    y_exact = mms.exact_y(x)   # phi  (unscaled y_bar*)
    p_exact = mms.exact_p(x)   # alpha * phi  (unscaled p_bar*)
    u_pred = -p_phys / mms.alpha
    u_exact = mms.exact_u(x)   # -phi

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        num = torch.sqrt(((a - b) ** 2).sum())
        den = torch.sqrt((b ** 2).sum()).clamp_min(1e-30)
        return (num / den).item()

    return {"l2_y": rel(y_phys, y_exact), "l2_p": rel(p_phys, p_exact), "l2_u": rel(u_pred, u_exact)}


# ---------------------------------------------------------------------------
# Parameter-list helpers (needed for gradient / Jacobian experiments)
# ---------------------------------------------------------------------------
def param_list(net_y: nn.Module, net_p: nn.Module) -> list[nn.Parameter]:
    return list(net_y.parameters()) + list(net_p.parameters())


def flat_grad(loss: torch.Tensor, params: Sequence[nn.Parameter], retain_graph: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True, create_graph=False)
    flats = []
    for g, p in zip(grads, params):
        flats.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(flats)


# ---------------------------------------------------------------------------
# Constants used across experiments
# ---------------------------------------------------------------------------
ALPHAS_STANDARD = (1.0, 1e-2, 1e-4, 1e-6, 1e-8)
SEEDS_STANDARD = (0, 1, 2, 3, 4)
N_INTERIOR = 2500
N_BDRY_PER_SIDE = 100  # 4 sides -> 400 boundary points total
