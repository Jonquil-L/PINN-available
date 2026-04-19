"""Experiment 1 — Jacobian conditioning (the core claim).

Goal: verify directly that
    kappa(J^T J) = Theta(alpha^-2)   for the unscaled system (1.4),
    kappa(J^T J) = Theta(alpha^-1)   for the scaled   system (1.5).

Procedure (follows the spec):
    At iter 0 (no training) and iter 1000 (1000 Adam steps at lr=1e-3):
        1. Assemble J in R^{2 N_r x P} by reverse-mode AD, one row per
           residual component.
        2. Compute sigma_max and sigma_min by RANDOMIZED SVD
           (scipy.sparse.linalg.svds with k=1, ARPACK-Lanczos bidiagonal).
        3. Record
               kappa(J)     = sigma_max / sigma_min
               kappa(J^T J) = (sigma_max / sigma_min)^2
Plot: log kappa(J^T J) vs log alpha with linear fits. Expected slopes -2
for (1.4), -1 for (1.5).

Net architecture: shallow dual-net MLP with 2 hidden layers x 20 units,
tanh, Xavier init. That gives P ~ 1.4e3, much smaller than the full net.

Interior sample size: N_r = 1000 (so J has 2000 rows x ~1.4e3 columns,
~11 MB in float32). The alpha-scaling law is a claim about the *spectrum*
of J and is unchanged by this choice of N_r. If you want N_r = 2500 as
elsewhere, bump N_R_EXP1 below (memory will ~double).

Output:
    results/exp1_conditioning_v2.csv
    results/exp1_conditioning_v2.png
"""
from __future__ import annotations

import csv
import math
import os
import platform
import sys
import time
from pathlib import Path

# Windows workaround for duplicated OpenMP runtime (libiomp5md.dll) in mixed
# scientific Python stacks. Must be set before importing torch/numpy/scipy.
if platform.system() == "Windows":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .common import (
        Formulation,
        ManufacturedSolution,
        SEEDS_STANDARD,
        build_networks,
        param_list,
        pick_device,
        residuals,
        sample_interior,
        total_loss,
    )
except ImportError:
    from common import (
        Formulation,
        ManufacturedSolution,
        SEEDS_STANDARD,
        build_networks,
        param_list,
        pick_device,
        residuals,
        sample_interior,
        total_loss,
    )

N_R_EXP1 = 1000
ITERS_SNAPSHOT = (0,)
LR_WARMUP = 1e-3

# Alpha sweep for Exp 1. We drop alpha = 1e-8 here because kappa(J^T J) ~ 1e16
# already at alpha = 1e-8 for (1.4), squaring to ~1e32-1e35, which exceeds the
# reliable range of float64 eigvalsh and poisons the log-log slope fit.
# ALPHAS_STANDARD in common.py is preserved for the other experiments.
ALPHAS_EXP1 = (1.0, 1e-2, 1e-4, 1e-6)

# Draw the iter-1000 snapshot as a second row of subplots. Off by default so
# the figure focuses on the uncontaminated, deterministic iter-0 result.
# The CSV still contains both snapshots regardless of this flag.
PLOT_ITER_1000 = False


# ---------------------------------------------------------------------------
# Jacobian assembly (reverse-mode row-by-row). Each row of J corresponds to
# a single residual component; one row = one backward pass w.r.t. params.
# ---------------------------------------------------------------------------
def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
    r1, r2 = residuals(net_y, net_p, x, mms, formulation)
    r = torch.cat([r1.reshape(-1), r2.reshape(-1)])
    params = param_list(net_y, net_p)
    M = r.numel()
    P = sum(p.numel() for p in params)
    J = torch.empty(M, P, device="cpu", dtype=torch.float64)  # keep master copy f64 on CPU
    for i in range(M):
        grads = torch.autograd.grad(r[i], params, retain_graph=True, allow_unused=True)
        flat = torch.cat(
            [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)]
        )
        J[i].copy_(flat.detach().to(dtype=torch.float64, device="cpu"))
    return J


# ---------------------------------------------------------------------------
# Extremal singular values via direct Gram-matrix eigendecomposition.
#
# Because J has shape (2 N_r, P) with 2 N_r << P (here 2000 << ~1.57e4),
# the smaller Gram factor A = J J^T is only 2 N_r x 2 N_r. Its spectrum
# equals the non-zero singular values squared of J; np.linalg.eigvalsh(A)
# returns them all at once in O((2 N_r)^3) flops -- both faster and more
# numerically stable than shift-invert Lanczos for the smallest singular
# value, which is what we actually care about for conditioning.
# ---------------------------------------------------------------------------
def sigma_extrema(J: torch.Tensor) -> tuple[float, float]:
    J_np = J.cpu().numpy()  # float64 already
    A = J_np @ J_np.T
    eigs = np.linalg.eigvalsh(A)
    eigs = np.clip(eigs, 0.0, None)
    sigma_max = float(np.sqrt(eigs[-1]))
    sigma_min = float(np.sqrt(max(eigs[0], 1e-300)))
    return sigma_max, sigma_min


# ---------------------------------------------------------------------------
# One point (alpha, formulation, seed, snapshot iter)
# ---------------------------------------------------------------------------
def one_point(alpha: float, formulation: Formulation, seed: int, device, snapshot_iter: int):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed, width=20, depth=2)  # shallow net: 2x20 tanh
    x = sample_interior(N_R_EXP1, seed=seed + 1000, device=device)

    if snapshot_iter > 0:
        opt = torch.optim.Adam(param_list(net_y, net_p), lr=LR_WARMUP)
        for _ in range(snapshot_iter):
            opt.zero_grad()
            loss = total_loss(net_y, net_p, x, mms, formulation)
            loss.backward()
            opt.step()

    # Assemble J, then compute extremal singular values.
    J = assemble_jacobian(net_y, net_p, x, mms, formulation)
    sigma_max, sigma_min = sigma_extrema(J)
    # Guard floor for log-plots.
    sigma_min = max(sigma_min, 1e-300)
    kappa_j = sigma_max / sigma_min
    # Avoid OverflowError for extreme conditioning at tiny alpha.
    max_float = sys.float_info.max
    kappa_jtj = max_float if kappa_j > math.sqrt(max_float) else kappa_j * kappa_j
    return {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "kappa_J": kappa_j,
        "kappa_JTJ": kappa_jtj,
        "J_shape": tuple(J.shape),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Running on device: {device} ({gpu_name})")
    else:
        print(f"Running on device: {device}")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    seeds = SEEDS_STANDARD  # 5 seeds per spec's common setup

    t0 = time.time()
    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    for snap in ITERS_SNAPSHOT:
        for formulation in formulations:
            for alpha in ALPHAS_EXP1:
                smax_arr, smin_arr, kJ_arr, kJTJ_arr = [], [], [], []
                for seed in seeds:
                    t_start = time.time()
                    r = one_point(alpha, formulation, seed, device, snap)
                    smax_arr.append(r["sigma_max"]); smin_arr.append(r["sigma_min"])
                    kJ_arr.append(r["kappa_J"]); kJTJ_arr.append(r["kappa_JTJ"])
                    print(
                        f"[iter={snap:4d}] {formulation:<8s} alpha={alpha:.0e} seed={seed}  "
                        f"J_shape={r['J_shape']}  "
                        f"sigma_max={r['sigma_max']:.3e}  sigma_min={r['sigma_min']:.3e}  "
                        f"kappa(J^T J)={r['kappa_JTJ']:.3e}  "
                        f"({time.time() - t_start:.1f}s)"
                    )
                rows.append(
                    {
                        "snapshot": snap,
                        "formulation": formulation,
                        "alpha": alpha,
                        "sigma_max_mean": float(np.mean(smax_arr)),
                        "sigma_max_std": float(np.std(smax_arr)),
                        "sigma_min_mean": float(np.mean(smin_arr)),
                        "sigma_min_std": float(np.std(smin_arr)),
                        "kappa_J_mean": float(np.mean(kJ_arr)),
                        "kappa_J_std": float(np.std(kJ_arr)),
                        "kappa_JTJ_mean": float(np.mean(kJTJ_arr)),
                        "kappa_JTJ_std": float(np.std(kJTJ_arr)),
                    }
                )
    print(f"Total time: {time.time() - t0:.1f}s")

    # CSV
    csv_path = out_dir / "exp1_conditioning_v2.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Plot. kappa(J) is the primary quantity (left panel); kappa(J^T J) is the
    # derived secondary (right panel). Show only iter=0 unless the user opts
    # in to the iter=1000 row via PLOT_ITER_1000.
    snaps_to_plot = list(ITERS_SNAPSHOT) if PLOT_ITER_1000 else [0]
    n_rows = len(snaps_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), squeeze=False)

    panel_spec = [
        ("kappa_J_mean",   "kappa_J_std",   r"$\kappa(J)=\sigma_{\max}/\sigma_{\min}$",
         {"unscaled": "-1", "scaled_raw": "-0.5"}),
        ("kappa_JTJ_mean", "kappa_JTJ_std", r"$\kappa(J^\top J)$",
         {"unscaled": "-2", "scaled_raw": "-1"}),
    ]

    for r_idx, snap in enumerate(snaps_to_plot):
        for c_idx, (metric_mean, metric_std, ylabel, expected) in enumerate(panel_spec):
            ax = axes[r_idx, c_idx]
            for formulation, marker, color in (
                ("unscaled", "o", "tomato"),
                ("scaled_raw", "s", "steelblue"),
            ):
                data = [r for r in rows if r["snapshot"] == snap and r["formulation"] == formulation]
                data.sort(key=lambda r: r["alpha"])
                a = np.array([r["alpha"] for r in data])
                mu = np.array([r[metric_mean] for r in data])
                sd = np.array([r[metric_std] for r in data])
                slope, intercept = np.polyfit(np.log10(a), np.log10(mu), 1)
                ax.errorbar(
                    a, mu, yerr=sd,
                    fmt=f"{marker}-", color=color, capsize=3,
                    label=f"{formulation} (fit slope {slope:+.2f}, expected {expected[formulation]})",
                )
                a_fit = np.geomspace(a.min(), a.max(), 50)
                ax.plot(a_fit, 10 ** (intercept + slope * np.log10(a_fit)),
                        color=color, lw=0.8, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.invert_xaxis()
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"snapshot: iter {snap}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

    fig.suptitle(
        r"Experiment 1 — Jacobian conditioning vs $\alpha$   "
        r"(expected slopes: $\kappa(J)$: $-1$ unscaled, $-0.5$ scaled_raw  ·  "
        r"$\kappa(J^\top J)$: $-2$ unscaled, $-1$ scaled_raw)"
    )
    plt.tight_layout()
    png = out_dir / "exp1_conditioning_v2.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
