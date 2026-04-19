"""Experiment 1 — Jacobian conditioning.

Goal: verify kappa(J^T J) = Theta(alpha^-2) for the unscaled system (1.4)
and Theta(alpha^-1) for the scaled system (1.5).

We use a *smaller* network here (width 20, depth 2) and a reduced interior
sample (N_r = 300) so that the full 2 N_r by P Jacobian fits in memory and
its singular values can be obtained by dense SVD. The scaling law in alpha
is an architectural statement and does not depend on the exact size.

Output:
    results/exp1_conditioning.csv
    results/exp1_conditioning.png
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import (
    ALPHAS_STANDARD,
    ManufacturedSolution,
    build_networks,
    flat_grad,
    param_list,
    pick_device,
    residuals,
    sample_interior,
    total_loss,
)


def assemble_jacobian(net_y, net_p, x, mms, formulation):
    """Build J in R^{2 N_r x P} by reverse-mode AD (one row per residual)."""
    r1, r2 = residuals(net_y, net_p, x, mms, formulation)
    r = torch.cat([r1.reshape(-1), r2.reshape(-1)])
    params = param_list(net_y, net_p)
    rows = []
    for i in range(r.numel()):
        grads = torch.autograd.grad(r[i], params, retain_graph=True, allow_unused=True)
        row = torch.cat(
            [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)]
        )
        rows.append(row.detach())
    return torch.stack(rows, dim=0)


def condition_number(J: torch.Tensor) -> float:
    """kappa(J^T J) = (sigma_max / sigma_min)^2, clamped below to avoid NaN."""
    # Use float64 on CPU for numerical reliability regardless of training dtype.
    J64 = J.detach().to(device="cpu", dtype=torch.float64)
    s = torch.linalg.svdvals(J64)
    s_max = s.max().item()
    s_min = s.min().clamp_min(1e-30).item()
    return (s_max / s_min) ** 2


def one_point(alpha: float, formulation: str, seed: int, device, iters_before_snapshot: int):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    # Small net keeps SVD tractable; the alpha-scaling we are measuring is
    # independent of net size.
    net_y, net_p = build_networks(device, seed=seed, width=20, depth=2)
    x = sample_interior(300, seed=seed + 1000, device=device)

    if iters_before_snapshot > 0:
        opt = torch.optim.Adam(param_list(net_y, net_p), lr=1e-3)
        for _ in range(iters_before_snapshot):
            opt.zero_grad()
            loss = total_loss(net_y, net_p, x, mms, formulation)
            loss.backward()
            opt.step()

    J = assemble_jacobian(net_y, net_p, x, mms, formulation)
    return condition_number(J)


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    snapshots = [0, 1000]
    formulations = ("unscaled", "scaled_raw")
    seeds = (0, 1, 2)  # conditioning is deterministic-ish; 3 seeds is plenty

    rows = []
    t0 = time.time()
    for snap in snapshots:
        for formulation in formulations:
            for alpha in ALPHAS_STANDARD:
                kappas = []
                for seed in seeds:
                    kappa = one_point(alpha, formulation, seed, device, snap)
                    kappas.append(kappa)
                kappas = np.array(kappas)
                rows.append(
                    {
                        "snapshot": snap,
                        "formulation": formulation,
                        "alpha": alpha,
                        "kappa_mean": kappas.mean(),
                        "kappa_std": kappas.std(),
                    }
                )
                print(
                    f"[iter={snap:4d}] {formulation:<8s} alpha={alpha:.0e} "
                    f"kappa(J^T J) = {kappas.mean():.3e} +/- {kappas.std():.3e}"
                )
    print(f"Total time: {time.time() - t0:.1f} s")

    # CSV
    csv_path = out_dir / "exp1_conditioning.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Plot (only the at-init snapshot, with linear fits in log-log)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, snap in zip(axes, snapshots):
        for formulation, marker, color in (("unscaled", "o", "tomato"), ("scaled_raw", "s", "steelblue")):
            data = [r for r in rows if r["snapshot"] == snap and r["formulation"] == formulation]
            data.sort(key=lambda r: r["alpha"])
            a = np.array([r["alpha"] for r in data])
            k = np.array([r["kappa_mean"] for r in data])
            kerr = np.array([r["kappa_std"] for r in data])
            # linear fit of log kappa = m log alpha + b
            m, b = np.polyfit(np.log10(a), np.log10(k), 1)
            ax.errorbar(a, k, yerr=kerr, fmt=f"{marker}-", color=color,
                        label=f"{formulation} (slope {m:.2f})", capsize=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\kappa(J^\top J)$")
        ax.set_title(f"Snapshot: iter {snap}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Experiment 1 — Jacobian conditioning vs alpha")
    plt.tight_layout()
    png = out_dir / "exp1_conditioning.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
