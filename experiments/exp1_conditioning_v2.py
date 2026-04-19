"""Experiment 1 — Jacobian conditioning (the core claim).

Goal: verify directly that
    kappa(J^T J) = Theta(alpha^-2)   for the unscaled system (1.4),
    kappa(J^T J) = Theta(alpha^-1)   for the scaled_raw system (1.5).

Equivalently, for kappa(J) = sigma_max / sigma_min:
    kappa(J) = Theta(alpha^-1)   for (1.4),
    kappa(J) = Theta(alpha^-0.5) for (1.5).

Key design choices
------------------
1. Use a SMALL network (depth=2, width=20).
   The alpha-scaling law is an algebraic property of the residual structure
   and holds for any network size. The full network (depth=4, width=50) has
   a base condition number ~10^{150} that overflows float64 and makes
   sigma_min indistinguishable from zero.

2. Work in LOG SPACE throughout.
   Store and plot log10(kappa); never materialise kappa(J^T J) as a float
   (it can exceed 10^{300} even with the small net at extreme alpha).

3. Use eigvalsh(J @ J.T) for exact singular values.
   The Gram matrix is at most (2*N_r) x (2*N_r) = 600x600 — cheap and
   deterministic. No ARPACK / randomised SVD needed.

4. Only plot the iter-0 snapshot.
   The conditioning is an algebraic property, not a training outcome.

Output
------
    results/exp1_conditioning_v2.csv
    results/exp1_conditioning_v2.png
"""
from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    Formulation,
    ManufacturedSolution,
    build_networks,
    param_list,
    pick_device,
    residuals,
    sample_interior,
    total_loss,
)

# ---- Experiment-specific constants ----------------------------------------

N_R_EXP1 = 1000           # interior collocation points (J has 2*N_R rows)

# Small network — see docstring for rationale.
EXP1_WIDTH = 50
EXP1_DEPTH = 4

# Alpha sweep.  Drop 1e-8: even with the small net, kappa(J^T J) ~ 10^{28}
# at alpha=1e-8, still representable but noisier.  Four points spanning 6
# decades give a clean slope fit.
ALPHAS_EXP1 = (1.0, 1e-2, 1e-4, 1e-6)

SEEDS_EXP1 = (0, 1, 2)   # nearly deterministic at init; 3 seeds suffice


# ---------------------------------------------------------------------------
# Jacobian assembly  (reverse-mode, row-by-row, float64 on CPU)
# ---------------------------------------------------------------------------
def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
    """Build J ∈ R^{2·N_r × P}, one backward pass per residual entry."""
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


# ---------------------------------------------------------------------------
# Extremal singular values via eigvalsh of the smaller Gram factor
# ---------------------------------------------------------------------------
def sigma_extrema(J: torch.Tensor) -> tuple[float, float]:
    """Return (sigma_max, sigma_min) from the eigenvalues of J J^T."""
    J_np = J.numpy()                        # already float64 on CPU
    A = J_np @ J_np.T                       # shape (2*N_r, 2*N_r)
    eigs = np.linalg.eigvalsh(A)            # ascending order
    eigs = np.clip(eigs, 0.0, None)         # clamp negative numerical noise
    sigma_max = float(np.sqrt(eigs[-1]))
    sigma_min = float(np.sqrt(max(eigs[0], 1e-300)))
    return sigma_max, sigma_min


# ---------------------------------------------------------------------------
# Clone networks + data to CPU float64 for spectral analysis
# ---------------------------------------------------------------------------
def _to_cpu_f64(net_y, net_p, x):
    ny = copy.deepcopy(net_y).cpu().to(torch.float64)
    np_ = copy.deepcopy(net_p).cpu().to(torch.float64)
    xx = x.detach().cpu().to(torch.float64)
    return ny, np_, xx


# ---------------------------------------------------------------------------
# One data point  →  log10(kappa_J)  and  log10(kappa_JTJ)
# ---------------------------------------------------------------------------
def one_point(
    alpha: float,
    formulation: Formulation,
    seed: int,
    device: torch.device,
    snapshot_iter: int,
) -> dict:
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=EXP1_WIDTH, depth=EXP1_DEPTH
    )
    x = sample_interior(N_R_EXP1, seed=seed + 1000, device=device)

    # Optional warm-up training (not used when snapshot_iter == 0)
    if snapshot_iter > 0:
        opt = torch.optim.Adam(param_list(net_y, net_p), lr=1e-3)
        for _ in range(snapshot_iter):
            opt.zero_grad()
            loss = total_loss(net_y, net_p, x, mms, formulation)
            loss.backward()
            opt.step()

    # Move to CPU float64 for spectral analysis
    ny64, np64, x64 = _to_cpu_f64(net_y, net_p, x)

    J = assemble_jacobian(ny64, np64, x64, mms, formulation)
    s_max, s_min = sigma_extrema(J)

    # ---- Work in log space: never form kappa_JTJ as a plain float ----
    log10_kJ   = math.log10(s_max) - math.log10(s_min)
    log10_kJTJ = 2.0 * log10_kJ

    return {
        "sigma_max":       s_max,
        "sigma_min":       s_min,
        "log10_kappa_J":   log10_kJ,
        "log10_kappa_JTJ": log10_kJTJ,
        "J_shape":         tuple(J.shape),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()

    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")

    for formulation in formulations:
        for alpha in ALPHAS_EXP1:
            log_kJ_list: list[float] = []
            log_kJTJ_list: list[float] = []
            smax_list: list[float] = []
            smin_list: list[float] = []

            for seed in SEEDS_EXP1:
                ts = time.time()
                r = one_point(alpha, formulation, seed, device,
                              snapshot_iter=0)
                log_kJ_list.append(r["log10_kappa_J"])
                log_kJTJ_list.append(r["log10_kappa_JTJ"])
                smax_list.append(r["sigma_max"])
                smin_list.append(r["sigma_min"])
                print(
                    f"{formulation:<10s}  alpha={alpha:.0e}  seed={seed}  "
                    f"J={r['J_shape']}  "
                    f"s_max={r['sigma_max']:.3e}  "
                    f"s_min={r['sigma_min']:.3e}  "
                    f"log10 k(J)={r['log10_kappa_J']:.2f}  "
                    f"log10 k(J^TJ)={r['log10_kappa_JTJ']:.2f}  "
                    f"({time.time()-ts:.1f}s)"
                )

            rows.append({
                "formulation":     formulation,
                "alpha":           alpha,
                "log10_kJ_mean":   float(np.mean(log_kJ_list)),
                "log10_kJ_std":    float(np.std(log_kJ_list)),
                "log10_kJTJ_mean": float(np.mean(log_kJTJ_list)),
                "log10_kJTJ_std":  float(np.std(log_kJTJ_list)),
                "sigma_max_mean":  float(np.mean(smax_list)),
                "sigma_min_mean":  float(np.mean(smin_list)),
            })

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # ---- CSV --------------------------------------------------------------
    csv_path = out_dir / "exp1_conditioning_v2.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # ---- Plot -------------------------------------------------------------
    #
    # Two panels side-by-side.
    #   Left:  log10 kappa(J)     vs  log10 alpha   (expected slopes -1, -0.5)
    #   Right: log10 kappa(J^TJ)  vs  log10 alpha   (expected slopes -2, -1)
    #
    # y-axis is LINEAR (showing log10 values) — this avoids ever
    # exponentiating back to kappa, which can overflow.
    #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panel_spec = [
        # (mean_key, std_key, y-label, {formulation: expected_slope})
        ("log10_kJ_mean",   "log10_kJ_std",
         r"$\log_{10}\;\kappa(J)$",
         {"unscaled": -1.0, "scaled_raw": -0.5}),
        ("log10_kJTJ_mean", "log10_kJTJ_std",
         r"$\log_{10}\;\kappa(J^\top J)$",
         {"unscaled": -2.0, "scaled_raw": -1.0}),
    ]

    styles = [
        ("unscaled",   "o", "tomato"),
        ("scaled_raw", "s", "steelblue"),
    ]

    for ax, (m_key, s_key, ylabel, expected) in zip(axes, panel_spec):
        for formulation, marker, color in styles:
            data = sorted(
                [r for r in rows if r["formulation"] == formulation],
                key=lambda r: r["alpha"],
            )
            log_a = np.array([np.log10(r["alpha"]) for r in data])
            mu    = np.array([r[m_key] for r in data])
            sd    = np.array([r[s_key] for r in data])

            # Linear fit:  log10(kappa) = slope * log10(alpha) + intercept
            slope, intercept = np.polyfit(log_a, mu, 1)
            exp_slope = expected[formulation]

            # Plot data points with error bars
            ax.errorbar(
                log_a, mu, yerr=sd,
                fmt=f"{marker}-", color=color, capsize=4, lw=1.5,
                label=(f"{formulation}  "
                       f"(fit {slope:+.2f},  expect {exp_slope:+.1f})"),
            )

            # Overlay the linear fit
            la_fit = np.linspace(log_a.min(), log_a.max(), 50)
            ax.plot(la_fit, intercept + slope * la_fit,
                    color=color, lw=1, ls=":")

        ax.set_xlabel(r"$\log_{10}\;\alpha$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Experiment 1 — Jacobian conditioning vs "
        r"$\alpha$  (depth-2, width-20 net, iter 0)"
        "\n"
        r"Expected slopes:  $\kappa(J)$: unscaled $-1$, scaled $-0.5$"
        r"  ·  $\kappa(J^\top\! J)$: unscaled $-2$, scaled $-1$",
        fontsize=11,
    )
    plt.tight_layout()
    png_path = out_dir / "exp1_conditioning_v2.png"
    plt.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()


