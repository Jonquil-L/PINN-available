"""Experiment 0 — data collection for the three principle-level figures.

Collects block Frobenius norms, sigma_max, and gradient ratio rho for a
sweep of (alpha, formulation, seed) triples. Writes a single CSV that
the three plotting scripts read.

Runtime: ~20-30 min on GPU, ~1-2 h on CPU with default SEEDS = range(10).
If you just want to test the pipeline, set SEEDS = (0, 1, 2).

Output:
    results/exp0_data.csv
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    Formulation,
    ManufacturedSolution,
    build_networks,
    flat_grad,
    loss_terms,
    param_list,
    pick_device,
    residuals,
    sample_interior,
)


# ---- Config ---------------------------------------------------------------

ALPHAS = (1.0, 1e-2, 1e-4, 1e-6)
SEEDS  = (0,1,2)      # bump from 3 -> 10 for publishable CIs

NET_WIDTH = 50
NET_DEPTH = 4

NR_JACOBIAN = 200                # each row = one backward pass
NR_GRADIENT = 2500               # only 2 backward passes total


# ---- Float64-on-CPU conversion for numerical stability --------------------

def _to_f64(net_y, net_p, x):
    ny  = copy.deepcopy(net_y).cpu().to(torch.float64)
    np_ = copy.deepcopy(net_p).cpu().to(torch.float64)
    xx  = x.detach().cpu().to(torch.float64)
    return ny, np_, xx


# ---- Jacobian assembly (row-by-row via reverse-mode AD) ------------------

def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
    """J in R^{2 N_r x P}, float64 on CPU."""
    r1, r2 = residuals(net_y, net_p, x, mms, formulation)
    r = torch.cat([r1.reshape(-1), r2.reshape(-1)])
    params = param_list(net_y, net_p)
    M = r.numel()
    P = sum(p.numel() for p in params)
    J = torch.empty(M, P, dtype=torch.float64)
    for i in range(M):
        grads = torch.autograd.grad(
            r[i], params, retain_graph=True, allow_unused=True
        )
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)
        ])
        J[i] = flat.detach().to(dtype=torch.float64, device="cpu")
    return J


def split_blocks(J, N_r, P_y):
    return (
        J[:N_r, :P_y],    # J11: eq1, y-net
        J[:N_r, P_y:],    # J12: eq1, p-net
        J[N_r:, :P_y],    # J21: eq2, y-net
        J[N_r:, P_y:],    # J22: eq2, p-net
    )


# ---- Measurements for one (alpha, formulation, seed) triple ---------------

def measure_one(alpha, formulation, seed, device):
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=NET_WIDTH, depth=NET_DEPTH
    )

    # Jacobian assembly on NR_JACOBIAN points.
    x_jac = sample_interior(NR_JACOBIAN, seed=seed + 1000, device=device)
    ny64, np64, x64 = _to_f64(net_y, net_p, x_jac)
    J = assemble_jacobian(ny64, np64, x64, mms, formulation)

    P_y = sum(p.numel() for p in ny64.parameters())
    J11, J12, J21, J22 = split_blocks(J, NR_JACOBIAN, P_y)

    block_norms = {
        "J11": torch.norm(J11).item(),
        "J12": torch.norm(J12).item(),
        "J21": torch.norm(J21).item(),
        "J22": torch.norm(J22).item(),
    }

    # sigma_max via the smaller Gram factor.
    J_np = J.numpy()
    m, n = J_np.shape
    G = J_np @ J_np.T if m <= n else J_np.T @ J_np
    eigs = np.linalg.eigvalsh(G)
    sigma_max = float(np.sqrt(max(eigs[-1], 0.0)))

    # Gradient ratio rho on NR_GRADIENT points (cheaper: only 2 backwards).
    x_grad = sample_interior(NR_GRADIENT, seed=seed + 2000, device=device)
    l1, l2 = loss_terms(net_y, net_p, x_grad, mms, formulation)
    params = param_list(net_y, net_p)
    g1 = flat_grad(l1, params, retain_graph=True)
    g2 = flat_grad(l2, params, retain_graph=False)
    rho = (g1.norm() / g2.norm().clamp_min(1e-30)).item()

    return {**block_norms, "sigma_max": sigma_max, "rho": rho}


# ---- Main sweep: one row per (alpha, formulation, seed) -------------------

def main():
    device = pick_device()
    print(f"Device: {device}")
    print(f"Seeds: {len(SEEDS)}  |  alphas: {len(ALPHAS)}  |  "
          f"total cells: {2 * len(ALPHAS) * len(SEEDS)}")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    rows = []
    t0 = time.time()

    for formulation in formulations:
        for alpha in ALPHAS:
            for seed in SEEDS:
                ts = time.time()
                r = measure_one(alpha, formulation, seed, device)
                row = {
                    "formulation": formulation,
                    "alpha": alpha,
                    "seed": seed,
                    **r,
                }
                rows.append(row)
                print(
                    f"{formulation:<10s} a={alpha:.0e} s={seed}  "
                    f"J11={r['J11']:.2e} J12={r['J12']:.2e} "
                    f"J21={r['J21']:.2e} J22={r['J22']:.2e}  "
                    f"s_max={r['sigma_max']:.2e}  "
                    f"rho={r['rho']:.2e}  ({time.time()-ts:.1f}s)"
                )

    print(f"\nTotal: {time.time()-t0:.1f}s")

    # Write long-format CSV: one row per (formulation, alpha, seed).
    csv_path = out_dir / "exp0_data.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")
    print()
    print("Now run the three plotting scripts:")
    print("  python exp0_plot_block_norms.py")
    print("  python exp0_plot_sigma_max.py")
    print("  python exp0_plot_rho.py")


if __name__ == "__main__":
    main()