"""Experiment 4 — Floating-point stability of the loss.

Claim: in float32, L_{(1.4)} loses its O(1)-magnitude terms (Laplacian, data
fit, source) to round-off when alpha is tiny, because the 1/alpha * p term
dominates and subtractions cancel against it with < machine-epsilon relative
precision. L_{(1.5)} keeps all terms O(1) because the scaled residual
balances each term by construction.

Setup: perturb the exact solution by Gaussian noise (std = 1e-3) on the
network outputs (we fake this by *subtracting* a small analytic perturbation
from the manufactured solution so we can still evaluate cleanly). Evaluate
each component of each residual and report its relative contribution to the
full residual L^2-norm, in both float32 and float64.

Since MPS has limited float64 support, the float64 path is forced to CPU.

Output:
    results/exp4_float_stability.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from .common import (
    ALPHAS_STANDARD,
    ManufacturedSolution,
    hard_bc,
    laplacian,
    pick_device,
    sample_interior,
)


def compute_term_contributions(alpha: float, formulation: str, dtype: torch.dtype, device: torch.device, noise: float = 1e-3):
    """Return dict: term -> L2^2 value (so they sum to ~ ||residual||^2)."""
    torch.manual_seed(7)
    mms = ManufacturedSolution(alpha)
    x = sample_interior(2500, seed=7, device=device, dtype=dtype)
    x = x.detach().clone().requires_grad_(True)

    # Manufactured y, p made as *closures of x* so autograd sees x -> y.
    pi = math.pi
    phi = torch.sin(pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
    # Add a small perturbation *that also depends on x* so Laplacian is well-defined
    gen = torch.Generator(device="cpu").manual_seed(13)
    coeffs = torch.randn(4, generator=gen).to(device=device, dtype=dtype)
    pert = noise * (
        coeffs[0] * torch.sin(2 * pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
        + coeffs[1] * torch.sin(pi * x[:, 0:1]) * torch.sin(2 * pi * x[:, 1:2])
        + coeffs[2] * torch.sin(3 * pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
        + coeffs[3] * torch.sin(pi * x[:, 0:1]) * torch.sin(3 * pi * x[:, 1:2])
    )
    y = phi + pert
    p = alpha * phi + alpha * pert  # keep p scaling consistent with exact p

    lap_y = laplacian(y, x)
    lap_p = laplacian(p, x)
    f = mms.source_f(x).to(dtype=dtype)
    ud = mms.prior_ud(x).to(dtype=dtype)
    yd = mms.target_yd(x).to(dtype=dtype)

    if formulation == "unscaled":
        t_lapy = (-lap_y) ** 2
        t_fud = (-(f + ud)) ** 2
        t_pinv = ((1.0 / alpha) * p) ** 2
        t_lapp = (-lap_p) ** 2
        t_y = (-y) ** 2
        t_yd = (yd) ** 2
        # Full residuals
        r1 = -lap_y - (f + ud) + (1.0 / alpha) * p
        r2 = -lap_p - y + yd
        out = {
            "||-Delta y||^2": t_lapy.mean().item(),
            "||f+u_d||^2": t_fud.mean().item(),
            "||p/alpha||^2": t_pinv.mean().item(),
            "||-Delta p||^2": t_lapp.mean().item(),
            "||y||^2": t_y.mean().item(),
            "||y_d||^2": t_yd.mean().item(),
            "||r1||^2": (r1 ** 2).mean().item(),
            "||r2||^2": (r2 ** 2).mean().item(),
        }
    else:  # scaled_raw: raw residuals of (1.5), NO division
        a12, a34, a14 = alpha ** 0.5, alpha ** 0.75, alpha ** 0.25
        r1 = -a12 * lap_y + p - a34 * (f + ud)
        r2 = -a12 * lap_p - y + a14 * yd
        out = {
            "||a^{1/2} Delta y||^2": ((a12 * lap_y) ** 2).mean().item(),
            "||p||^2": (p ** 2).mean().item(),
            "||a^{3/4}(f+u_d)||^2": ((a34 * (f + ud)) ** 2).mean().item(),
            "||a^{1/2} Delta p||^2": ((a12 * lap_p) ** 2).mean().item(),
            "||y||^2": (y ** 2).mean().item(),
            "||a^{1/4} y_d||^2": ((a14 * yd) ** 2).mean().item(),
            "||r1||^2": (r1 ** 2).mean().item(),
            "||r2||^2": (r2 ** 2).mean().item(),
        }
    return out


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    device_f32 = pick_device()
    device_f64 = torch.device("cpu")  # reliable float64 everywhere

    rows = []
    for alpha in ALPHAS_STANDARD:
        for formulation in ("unscaled", "scaled_raw"):
            terms32 = compute_term_contributions(alpha, formulation, torch.float32, device_f32)
            terms64 = compute_term_contributions(alpha, formulation, torch.float64, device_f64)
            for precision, terms in (("float32", terms32), ("float64", terms64)):
                row = {"alpha": alpha, "formulation": formulation, "precision": precision}
                row.update({k: v for k, v in terms.items()})
                rows.append(row)

    # Union of all keys
    keys = sorted({k for r in rows for k in r.keys()})
    # Put meta first
    priority = ["alpha", "formulation", "precision"]
    keys = priority + [k for k in keys if k not in priority]

    csv_path = out_dir / "exp4_float_stability.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {csv_path}")

    # Console table
    for r in rows:
        print(f"alpha={r['alpha']:.0e} {r['formulation']:<8s} {r['precision']:<7s} "
              f"||r1||^2={r.get('||r1||^2', float('nan')):.3e} "
              f"||r2||^2={r.get('||r2||^2', float('nan')):.3e}")


if __name__ == "__main__":
    main()
