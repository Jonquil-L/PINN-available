"""Experiment 3 — Maximum stable Adam learning rate.

Claim: the largest eta for which training remains stable scales as alpha^2
for (1.4) but is alpha-independent for (1.5).

Stability criterion:
    * loss at iter 20000 is BELOW its value at iter 100, AND
    * 500-step moving average is monotone non-increasing over the last 5 windows.
A run that NaNs or Inf's is counted as unstable.

Output:
    results/exp3_eta_max.csv
    results/exp3_eta_max.png
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .common import (
        ALPHAS_STANDARD,
        Formulation,
        ManufacturedSolution,
        build_networks,
        param_list,
        pick_device,
        sample_interior,
        total_loss,
    )
except ImportError:
    # Allow direct execution: python experiments/exp3_stable_lr.py
    from common import (  # type: ignore[no-redef]
        ALPHAS_STANDARD,
        Formulation,
        ManufacturedSolution,
        build_networks,
        param_list,
        pick_device,
        sample_interior,
        total_loss,
    )

ITERS = 20_000
PRINT_EVERY = 1_000
LR_SWEEP = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)


def run_trial(alpha: float, formulation: Formulation, lr: float, seed: int, device) -> bool:
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    opt = torch.optim.Adam(param_list(net_y, net_p), lr=lr)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    loss_hist = []
    for it in range(ITERS):
        opt.zero_grad()
        loss = total_loss(net_y, net_p, x, mms, formulation)
        if it % PRINT_EVERY == 0:
            print(
                f"[{formulation} a={alpha:.0e} lr={lr:.0e} seed={seed}] "
                f"iter {it:5d}/{ITERS} loss={loss.item():.3e}"
            )
        if not torch.isfinite(loss):
            return False
        loss.backward()
        # Skip update if any grad is NaN/Inf (treated as divergence)
        bad = False
        for p in param_list(net_y, net_p):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = True
                break
        if bad:
            return False
        opt.step()
        loss_hist.append(loss.item())

    arr = np.array(loss_hist)
    if not np.isfinite(arr).all():
        return False
    if arr[-1] >= arr[100]:
        return False
    # Monotone moving-average check over last 5 windows of 500
    window = 500
    tail = arr[-5 * window:]
    means = [tail[i * window : (i + 1) * window].mean() for i in range(5)]
    return all(means[i + 1] <= means[i] * 1.05 for i in range(4))  # 5% slack


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    formulations: tuple[Formulation, Formulation] = ("unscaled", "scaled")
    for formulation in formulations:
        for alpha in ALPHAS_STANDARD:
            eta_max = 0.0
            for lr in sorted(LR_SWEEP, reverse=True):
                ok = run_trial(alpha, formulation, lr, seed=0, device=device)
                print(f"{formulation:<8s} alpha={alpha:.0e} lr={lr:.0e} -> {'stable' if ok else 'unstable'}")
                if ok:
                    eta_max = lr
                    break  # biggest stable lr found
            rows.append({"formulation": formulation, "alpha": alpha, "eta_max": eta_max})

    csv_path = out_dir / "exp3_eta_max.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for formulation, marker, color in (("unscaled", "o", "tomato"), ("scaled", "s", "steelblue")):
        data = [r for r in rows if r["formulation"] == formulation]
        data.sort(key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in data])
        e = np.array([r["eta_max"] for r in data])
        # log-log fit (skip zeros)
        mask = e > 0
        if mask.sum() >= 2:
            m, b = np.polyfit(np.log10(a[mask]), np.log10(e[mask]), 1)
            label = f"{formulation} (slope {m:.2f})"
        else:
            label = f"{formulation}"
        ax.loglog(a, np.where(e > 0, e, np.nan), f"{marker}-", color=color, label=label)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\eta_{\max}$ (largest stable LR)")
    ax.set_title("Experiment 3 — max stable learning rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    png = out_dir / "exp3_eta_max.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
