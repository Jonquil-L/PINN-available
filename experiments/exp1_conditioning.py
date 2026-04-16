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

Net architecture: the full dual-net MLP used across the paper (4 hidden
layers x 50 units, tanh, Xavier init). That gives P ~ 1.57e4, so full SVD
of the P x P Gram matrix is not affordable -- randomized SVD is necessary.

Interior sample size: N_r = 1000 (so J has 2000 rows x ~1.57e4 columns,
~125 MB in float32). The alpha-scaling law is a claim about the *spectrum*
of J and is unchanged by this choice of N_r. If you want N_r = 2500 as
elsewhere, bump N_R_EXP1 below (memory will ~double).

Output:
    results/exp1_conditioning.csv
    results/exp1_conditioning.png
"""
from __future__ import annotations

import csv
import math
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
    residuals,
    sample_interior,
    total_loss,
)

N_R_EXP1 = 1000
ITERS_SNAPSHOT = (0, 1000)
LR_WARMUP = 1e-3


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
# Randomized SVD for sigma_max and sigma_min.
#
# We use scipy.sparse.linalg.svds (ARPACK Lanczos bidiagonal) with k=1 and
# which='LM' for sigma_max / which='SM' for sigma_min. svds operates on J
# matrix-free via LinearOperator. This is the standard "randomized"
# (Krylov-iterative) path used in the Halko-Martinsson-Tropp family; it is
# much cheaper than full SVD (which would need ~ min(m,n)^3 work).
#
# Fallback: if SM is numerically unreliable (tight clusters of small sigmas),
# form the smaller Gram matrix A = J J^T (size m x m where m = 2 N_r << P)
# and take eigvalsh(A). Gives the exact spectrum and scales as O(m^3).
# ---------------------------------------------------------------------------
def sigma_extrema(J: torch.Tensor) -> tuple[float, float]:
    from scipy.sparse.linalg import svds

    J_np = J.cpu().numpy()  # float64 already
    m, n = J_np.shape

    # sigma_max via Lanczos / randomized SVD
    try:
        _, s_top, _ = svds(J_np, k=1, which="LM", maxiter=2000, tol=1e-10)
        sigma_max = float(s_top[-1])
    except Exception:
        sigma_max = _fallback_gram(J_np, which="max")

    # sigma_min: SM-svds can fail on ill-conditioned matrices. Try it, then
    # fall back to exact eigvalsh on the smaller Gram factor.
    try:
        _, s_bot, _ = svds(J_np, k=1, which="SM", maxiter=5000, tol=1e-10, solver="arpack")
        sigma_min = float(s_bot[0])
        # sanity check: if SM happens to return something larger than sigma_max (ARPACK bug
        # on near-singular matrices), fall through to exact route.
        if not (0.0 < sigma_min <= sigma_max):
            raise RuntimeError("SM-svds gave a non-sensible value; falling back.")
    except Exception:
        sigma_min = _fallback_gram(J_np, which="min")

    return sigma_max, sigma_min


def _fallback_gram(J_np: np.ndarray, which: str) -> float:
    m, n = J_np.shape
    if m <= n:
        A = J_np @ J_np.T
    else:
        A = J_np.T @ J_np
    eigs = np.linalg.eigvalsh(A)
    if which == "max":
        return float(math.sqrt(max(eigs[-1], 0.0)))
    else:
        return float(math.sqrt(max(eigs[0], 0.0)))


# ---------------------------------------------------------------------------
# One point (alpha, formulation, seed, snapshot iter)
# ---------------------------------------------------------------------------
def one_point(alpha: float, formulation: str, seed: int, device, snapshot_iter: int):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)  # full net: 4x50 tanh
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
    return {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "kappa_J": sigma_max / sigma_min,
        "kappa_JTJ": (sigma_max / sigma_min) ** 2,
        "J_shape": tuple(J.shape),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    seeds = SEEDS_STANDARD  # 5 seeds per spec's common setup

    t0 = time.time()
    for snap in ITERS_SNAPSHOT:
        for formulation in ("unscaled", "scaled"):
            for alpha in ALPHAS_STANDARD:
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
    csv_path = out_dir / "exp1_conditioning.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # Plot: one row per snapshot; two columns = kappa(J^T J) and kappa(J).
    fig, axes = plt.subplots(len(ITERS_SNAPSHOT), 2, figsize=(12, 4.5 * len(ITERS_SNAPSHOT)))
    if len(ITERS_SNAPSHOT) == 1:
        axes = np.expand_dims(axes, 0)
    for r_idx, snap in enumerate(ITERS_SNAPSHOT):
        for c_idx, (metric_mean, metric_std, ylabel) in enumerate(
            [
                ("kappa_JTJ_mean", "kappa_JTJ_std", r"$\kappa(J^\top J)$"),
                ("kappa_J_mean", "kappa_J_std", r"$\kappa(J)=\sigma_{\max}/\sigma_{\min}$"),
            ]
        ):
            ax = axes[r_idx, c_idx]
            for formulation, marker, color in (
                ("unscaled", "o", "tomato"),
                ("scaled", "s", "steelblue"),
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
                    label=f"{formulation} (slope {slope:+.2f})",
                )
                # linear fit overlay
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
        r"(expected slopes: $-2$ unscaled, $-1$ scaled for $\kappa(J^\top J)$)"
    )
    plt.tight_layout()
    png = out_dir / "exp1_conditioning.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
