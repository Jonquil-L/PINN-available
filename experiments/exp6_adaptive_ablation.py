"""Experiment 6 — Can adaptive loss weights rescue (1.4)?

We compare three methods:
    (1) unscaled (1.4) with equal weights              -- baseline from Exp 5
    (2) unscaled (1.4) + Wang et al. Algorithm 1       -- adaptive weighting
    (3) scaled   (1.5) with equal weights              -- our claim

Wang et al. "learning-rate annealing" update (Algorithm 1):
    Anchor lambda_1 := 1. Every K steps, for i != 1, compute
        lambda_hat_i = max_phi |grad_phi L_1| / mean_phi |grad_phi L_i|
        lambda_i <- (1 - alpha) * lambda_i + alpha * lambda_hat_i
    and use L = sum_i lambda_i L_i.

Output:
    results/exp6_ablation.csv
    results/exp6_ablation.png
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
    loss_terms,
    param_list,
    pick_device,
    relative_l2_errors,
    sample_interior,
    total_loss,
)

ITERS = 20_000
LR = 1e-3
UPDATE_EVERY = 10
BETA = 0.9  # lambda momentum


def _grad_stats_per_loss(loss: torch.Tensor, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    all_abs = []
    for g, p in zip(grads, params):
        if g is None:
            continue
        all_abs.append(g.detach().abs().reshape(-1))
    flat = torch.cat(all_abs) if all_abs else torch.tensor([0.0])
    return flat.max().item(), flat.mean().item()


def train_adaptive(alpha, seed, device):
    """Train (1.4) with Wang et al. Algorithm 1."""
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    params = param_list(net_y, net_p)
    opt = torch.optim.Adam(params, lr=LR)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    lam1, lam2 = 1.0, 1.0

    for it in range(ITERS):
        opt.zero_grad()
        l1, l2 = loss_terms(net_y, net_p, x, mms, "unscaled")
        if not torch.isfinite(l1) or not torch.isfinite(l2):
            return None

        if it % UPDATE_EVERY == 0 and it > 0:
            max1, _mean1 = _grad_stats_per_loss(l1, params)
            _max2, mean2 = _grad_stats_per_loss(l2, params)
            lam1 = 1.0  # anchor equation 1
            if mean2 > 0:
                lam2_hat = max1 / mean2
                lam2 = (1 - BETA) * lam2_hat + BETA * lam2

        loss = lam1 * l1 + lam2 * l2
        if not torch.isfinite(loss):
            return None
        loss.backward()
        opt.step()

    return relative_l2_errors(net_y, net_p, mms, device, formulation="unscaled")


def train_plain(alpha, formulation, seed, device):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    opt = torch.optim.Adam(param_list(net_y, net_p), lr=LR)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    for _ in range(ITERS):
        opt.zero_grad()
        loss = total_loss(net_y, net_p, x, mms, formulation)
        if not torch.isfinite(loss):
            return None
        loss.backward()
        opt.step()

    return relative_l2_errors(net_y, net_p, mms, device, formulation=formulation)


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    methods = ("unscaled", "unscaled+adaptive", "scaled_raw")
    rows = []
    t0 = time.time()
    for method in methods:
        for alpha in ALPHAS_STANDARD:
            per_seed = {"l2_y": [], "l2_p": [], "l2_u": []}
            for seed in SEEDS_STANDARD:
                if method == "unscaled":
                    errs = train_plain(alpha, "unscaled", seed, device)
                elif method == "scaled_raw":
                    errs = train_plain(alpha, "scaled_raw", seed, device)
                else:
                    errs = train_adaptive(alpha, seed, device)
                if errs is None:
                    errs = {"l2_y": float("nan"), "l2_p": float("nan"), "l2_u": float("nan")}
                for k in per_seed:
                    per_seed[k].append(errs[k])
                print(f"{method:<20s} alpha={alpha:.0e} seed={seed} -> "
                      f"L2_y={errs['l2_y']:.3e} L2_p={errs['l2_p']:.3e} L2_u={errs['l2_u']:.3e}")
            row = {"method": method, "alpha": alpha}
            for k in per_seed:
                vals = np.array(per_seed[k])
                row[f"{k}_mean"] = np.nanmean(vals)
                row[f"{k}_std"] = np.nanstd(vals)
            rows.append(row)
    print(f"Total time: {time.time() - t0:.1f} s")

    csv_path = out_dir / "exp6_ablation.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    style = {
        "unscaled":           ("o", "tomato"),
        "unscaled+adaptive":  ("^", "darkgreen"),
        "scaled_raw":         ("s", "steelblue"),
    }
    for ax, var, label in zip(axes, ("l2_y", "l2_p", "l2_u"), (r"$\bar y$", r"$\bar p$", r"$\bar u$")):
        for method in methods:
            data = [r for r in rows if r["method"] == method]
            data.sort(key=lambda r: r["alpha"])
            a = np.array([r["alpha"] for r in data])
            m = np.array([r[f"{var}_mean"] for r in data])
            s = np.array([r[f"{var}_std"] for r in data])
            marker, color = style[method]
            ax.errorbar(a, m, yerr=s, fmt=f"{marker}-", color=color, label=method, capsize=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(f"relative $L^2$ error of {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Experiment 6 — adaptive weighting on (1.4) vs scaled (1.5)")
    plt.tight_layout()
    png = out_dir / "exp6_ablation.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
