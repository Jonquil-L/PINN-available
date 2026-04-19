"""Experiment 1 — Jacobian conditioning (the core claim).

Goal: verify directly that
    kappa(J^T J) = Theta(alpha^-2)   for the unscaled system (1.4),
    kappa(J^T J) = Theta(alpha^-1)   for the scaled_raw system (1.5).

Equivalently, for kappa(J) = sigma_max / sigma_min:
    kappa(J) = Theta(alpha^-1)   for (1.4),
    kappa(J) = Theta(alpha^-0.5) for (1.5).

Key design choices
------------------
1. Use a SMALL network (depth=2, width=20, total P ~ 1002 parameters).
   The alpha-scaling law is an algebraic property of the residual structure
   and holds for any network size.

2. Use enough collocation points so that the system is OVER-DETERMINED:
   2 * N_r > P.  With N_r = 1000, J has shape (2000, 1002).
   This ensures that sigma_min is a genuine singular value of J, not an
   artifact of the network's null space.  When the system is under-
   determined (2*N_r < P), the Gram matrix J @ J^T has zero eigenvalues
   from the null space, and sigma_min = sqrt(noise) ~ 10^{-160}, giving
   a meaningless condition number of 10^{150+}.

3. Use the CORRECT Gram factor for eigendecomposition:
   - Over-determined (m >= n):  A = J^T J  (n x n), all n eigenvalues real
   - Under-determined (m < n):  A = J J^T  (m x m), has zero eigenvalues

4. Work in LOG SPACE throughout: store log10(kappa), never materialise
   kappa(J^T J) as a float.

5. Only compute at iter-0 (algebraic property, not training outcome).

Output:
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

# Network size: small enough that base kappa is manageable.
EXP1_WIDTH = 20
EXP1_DEPTH = 2
# Total params P = 2 * [(2*20+20) + (20*20+20) + (20*1+1)] = 2 * 501 = 1002

# Collocation points: MUST satisfy 2*N_r > P for over-determined system.
# 2*1000 = 2000 > 1002 = P.  ✓
N_R_EXP1 = 1000

# Alpha sweep
ALPHAS_EXP1 = (1.0, 1e-2, 1e-4, 1e-6)

# Seeds (nearly deterministic at init; 3 seeds suffice)
SEEDS_EXP1 = (0, 1, 2)


# ---------------------------------------------------------------------------
# Jacobian assembly  (reverse-mode, row-by-row, float64 on CPU)
# ---------------------------------------------------------------------------
def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
    """Build J in R^{2*N_r x P}, one backward pass per residual entry."""
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
# Extremal singular values via eigendecomposition of the CORRECT Gram factor.
#
# For J of shape (m, n):
#   if m >= n (over-determined):  A = J^T J  (n x n)
#       All n eigenvalues are squared singular values — sigma_min is real.
#   if m < n  (under-determined): A = J J^T  (m x m)
#       Has zero eigenvalues from null space — sigma_min is garbage.
# ---------------------------------------------------------------------------
def sigma_extrema(J: torch.Tensor) -> tuple[float, float]:
    J_np = J.numpy()
    m, n = J_np.shape

    if m >= n:
        # Over-determined: J^T J gives all n genuine squared singular values
        A = J_np.T @ J_np     # shape (n, n)
    else:
        # Under-determined: J J^T — sigma_min will be unreliable
        A = J_np @ J_np.T     # shape (m, m)

    eigs = np.linalg.eigvalsh(A)       # ascending order
    eigs = np.clip(eigs, 0.0, None)    # clamp negative numerical noise

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
# One data point -> log10(kappa_J), log10(kappa_JTJ)
# ---------------------------------------------------------------------------
def one_point(
    alpha: float,
    formulation: Formulation,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=EXP1_WIDTH, depth=EXP1_DEPTH
    )
    x = sample_interior(N_R_EXP1, seed=seed + 1000, device=device)

    # Move to CPU float64 for spectral analysis
    ny64, np64, x64 = _to_cpu_f64(net_y, net_p, x)

    J = assemble_jacobian(ny64, np64, x64, mms, formulation)
    m, n = J.shape
    status = "OVER-determined ✓" if m >= n else "UNDER-determined ✗"
    print(f"    J shape: ({m}, {n})  {status}")

    s_max, s_min = sigma_extrema(J)

    # Work in log space — never form kappa^2 as a float
    log10_kJ   = math.log10(s_max) - math.log10(s_min)
    log10_kJTJ = 2.0 * log10_kJ

    return {
        "sigma_max":       s_max,
        "sigma_min":       s_min,
        "log10_kappa_J":   log10_kJ,
        "log10_kappa_JTJ": log10_kJTJ,
        "J_shape":         (m, n),
    }


# ---------------------------------------------------------------------------
# Main
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
            log_kJ_list:   list[float] = []
            log_kJTJ_list: list[float] = []
            smax_list:     list[float] = []
            smin_list:     list[float] = []

            for seed in SEEDS_EXP1:
                ts = time.time()
                r = one_point(alpha, formulation, seed, device)
                log_kJ_list.append(r["log10_kappa_J"])
                log_kJTJ_list.append(r["log10_kappa_JTJ"])
                smax_list.append(r["sigma_max"])
                smin_list.append(r["sigma_min"])
                print(
                    f"  {formulation:<10s}  alpha={alpha:.0e}  seed={seed}  "
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
    # Two panels: log10 kappa(J) and log10 kappa(J^T J) vs log10 alpha.
    # y-axis is LINEAR (log10 values) — no exponentiation, no overflow.

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panel_spec = [
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

            # Linear fit: log10(kappa) = slope * log10(alpha) + intercept
            slope, intercept = np.polyfit(log_a, mu, 1)
            exp_slope = expected[formulation]

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
        r"$\alpha$  (depth-2, width-20 net, $N_r$=1000, iter 0)"
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


