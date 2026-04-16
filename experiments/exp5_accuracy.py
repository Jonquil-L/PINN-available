"""Experiment 5 — End-to-end accuracy vs alpha.

Bottom line: with everything fixed (Adam, lr=1e-3, 20k iterations, same
architecture, no adaptive weighting), report the relative L2 errors of
y_phi, p_phi, and u_phi = -p_phi/alpha against the manufactured solution.

Expected:
    * Unscaled (1.4) error blows up once alpha <= ~1e-3.
    * Scaled (1.5) error stays near 1e-3 across the full alpha range.

Output:
    results/exp5_accuracy.csv
    results/exp5_accuracy.png
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
    SEEDS_STANDARD,
    build_networks,
    param_list,
    pick_device,
    relative_l2_errors,
    sample_interior,
    total_loss,
)

ITERS = 20_000
LR = 1e-3


def train_once(alpha: float, formulation: str, seed: int, device):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    opt = torch.optim.Adam(param_list(net_y, net_p), lr=LR)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    for it in range(ITERS):
        opt.zero_grad()
        loss = total_loss(net_y, net_p, x, mms, formulation)
        if not torch.isfinite(loss):
            return None  # diverged
        loss.backward()
        opt.step()

    errs = relative_l2_errors(net_y, net_p, mms, device)
    return errs


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for formulation in ("unscaled", "scaled"):
        for alpha in ALPHAS_STANDARD:
            per_seed = {"l2_y": [], "l2_p": [], "l2_u": []}
            for seed in SEEDS_STANDARD:
                errs = train_once(alpha, formulation, seed, device)
                if errs is None:
                    errs = {"l2_y": float("nan"), "l2_p": float("nan"), "l2_u": float("nan")}
                for k in per_seed:
                    per_seed[k].append(errs[k])
                print(f"{formulation:<8s} alpha={alpha:.0e} seed={seed} -> "
                      f"L2_y={errs['l2_y']:.3e} L2_p={errs['l2_p']:.3e} L2_u={errs['l2_u']:.3e}")
            row = {"formulation": formulation, "alpha": alpha}
            for k in per_seed:
                vals = np.array(per_seed[k])
                row[f"{k}_mean"] = np.nanmean(vals)
                row[f"{k}_std"] = np.nanstd(vals)
            rows.append(row)
    print(f"Total time: {time.time() - t0:.1f} s")

    csv_path = out_dir / "exp5_accuracy.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, var, label in zip(axes, ("l2_y", "l2_p", "l2_u"), (r"$\bar y$", r"$\bar p$", r"$\bar u$")):
        for formulation, marker, color in (("unscaled", "o", "tomato"), ("scaled", "s", "steelblue")):
            data = [r for r in rows if r["formulation"] == formulation]
            data.sort(key=lambda r: r["alpha"])
            a = np.array([r["alpha"] for r in data])
            m = np.array([r[f"{var}_mean"] for r in data])
            s = np.array([r[f"{var}_std"] for r in data])
            ax.errorbar(a, m, yerr=s, fmt=f"{marker}-", color=color,
                        label=formulation, capsize=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(f"relative $L^2$ error of {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Experiment 5 — accuracy vs alpha (Adam, lr=1e-3, 20k iter, 5 seeds)")
    plt.tight_layout()
    png = out_dir / "exp5_accuracy.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
