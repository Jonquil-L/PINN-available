"""Experiment 1 — Jacobian structure analysis (redesigned).

Three complementary measurements, all numerically robust (no sigma_min):

Panel (a) — Block Frobenius norms
    J has 2x2 block structure: rows = {eq1, eq2}, cols = {y_net, p_net}.
    (1.4): J_12 ~ alpha^{-1}, others O(1).
    (1.5): J_11, J_22 ~ alpha^{1/2}, J_12, J_21 ~ O(1).

Panel (b) — sigma_max(J) vs alpha
    The largest singular value controls the Lipschitz constant of grad L,
    hence the maximum stable learning rate (Experiment 3).
    (1.4): sigma_max ~ alpha^{-1}  (slope -1)
    (1.5): sigma_max ~ const       (slope ~0)

Panel (c) — Gradient ratio rho at initialisation
    rho = ||grad L_eq1|| / ||grad L_eq2||
    (1.4): rho ~ alpha^{-1}  (slope -1)  — the Wang et al. pathology
    (1.5): rho ~ O(1)        (slope ~0)  — balanced

All use the full paper network (4x50 tanh). No sigma_min is needed
anywhere — this is the key insight after discovering that sigma_min
of a PINN Jacobian is always at the float64 noise floor regardless
of alpha or network size.

Output:
    results/exp1_analysis.csv
    results/exp1_analysis.png
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
    flat_grad,
    loss_terms,
    param_list,
    pick_device,
    residuals,
    sample_interior,
)

# ---- Constants ------------------------------------------------------------

ALPHAS = (1.0, 1e-2, 1e-4, 1e-6)
SEEDS  = (0, 1, 2)

# Full paper network
NET_WIDTH = 50
NET_DEPTH = 4

# Jacobian assembly uses fewer points (each row = one backward pass)
NR_JACOBIAN = 200   # J is (400, ~15700) — ~400 backward passes

# Gradient ratio uses standard sample size (only 2 backward passes total)
NR_GRADIENT = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_f64(net_y, net_p, x):
    ny = copy.deepcopy(net_y).cpu().to(torch.float64)
    np_ = copy.deepcopy(net_p).cpu().to(torch.float64)
    xx = x.detach().cpu().to(torch.float64)
    return ny, np_, xx


def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
    """J in R^{2*N_r x P}, float64 on CPU."""
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
    """Split J into 4 blocks: (eq, sub-network)."""
    return (
        J[:N_r, :P_y],    # J11: eq1, y-net
        J[:N_r, P_y:],    # J12: eq1, p-net
        J[N_r:, :P_y],    # J21: eq2, y-net
        J[N_r:, P_y:],    # J22: eq2, p-net
    )


# ---------------------------------------------------------------------------
# Measurements for one (alpha, formulation, seed) triple
# ---------------------------------------------------------------------------
def measure_one(alpha, formulation, seed, device):
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=NET_WIDTH, depth=NET_DEPTH
    )

    # ---- Panels (a) and (b): Jacobian block norms + sigma_max ----
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

    # sigma_max from the smaller Gram factor
    J_np = J.numpy()
    m, n = J_np.shape
    if m <= n:
        G = J_np @ J_np.T
    else:
        G = J_np.T @ J_np
    eigs = np.linalg.eigvalsh(G)
    sigma_max = float(np.sqrt(max(eigs[-1], 0.0)))

    # ---- Panel (c): Gradient ratio rho ----
    x_grad = sample_interior(NR_GRADIENT, seed=seed + 2000, device=device)
    l1, l2 = loss_terms(net_y, net_p, x_grad, mms, formulation)
    params = param_list(net_y, net_p)
    g1 = flat_grad(l1, params, retain_graph=True)
    g2 = flat_grad(l2, params, retain_graph=False)
    rho = (g1.norm() / g2.norm().clamp_min(1e-30)).item()

    return {**block_norms, "sigma_max": sigma_max, "rho": rho}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    rows = []
    t0 = time.time()

    for formulation in formulations:
        for alpha in ALPHAS:
            seed_data: dict[str, list[float]] = {
                k: [] for k in
                ("J11", "J12", "J21", "J22", "sigma_max", "rho")
            }
            for seed in SEEDS:
                ts = time.time()
                r = measure_one(alpha, formulation, seed, device)
                for k in seed_data:
                    seed_data[k].append(r[k])
                print(
                    f"{formulation:<10s} a={alpha:.0e} s={seed}  "
                    f"J11={r['J11']:.2e} J12={r['J12']:.2e} "
                    f"J21={r['J21']:.2e} J22={r['J22']:.2e}  "
                    f"s_max={r['sigma_max']:.2e}  rho={r['rho']:.2e}  "
                    f"({time.time()-ts:.1f}s)"
                )

            row: dict = {"formulation": formulation, "alpha": alpha}
            for k, vals in seed_data.items():
                arr = np.array(vals)
                row[f"{k}_mean"] = float(np.mean(arr))
                row[f"{k}_std"]  = float(np.std(arr))
            # Log10 versions for slope fitting
            row["log10_sigma_max_mean"] = float(
                np.mean(np.log10(seed_data["sigma_max"])))
            row["log10_sigma_max_std"] = float(
                np.std(np.log10(seed_data["sigma_max"])))
            row["log10_rho_mean"] = float(
                np.mean(np.log10(seed_data["rho"])))
            row["log10_rho_std"] = float(
                np.std(np.log10(seed_data["rho"])))
            rows.append(row)

    print(f"\nTotal: {time.time()-t0:.1f}s")

    # ---- CSV --------------------------------------------------------------
    csv_path = out_dir / "exp1_analysis.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ---- Plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sty = {"unscaled": ("o", "tomato"), "scaled_raw": ("s", "steelblue")}

    # ---- (a) Block Frobenius norms ----------------------------------------
    ax = axes[0]
    bk_info = [
        ("J11", r"$\|J_{11}\|$ eq1,y", "C0"),
        ("J12", r"$\|J_{12}\|$ eq1,p", "C1"),
        ("J21", r"$\|J_{21}\|$ eq2,y", "C2"),
        ("J22", r"$\|J_{22}\|$ eq2,p", "C3"),
    ]
    for formulation in formulations:
        data = sorted(
            [r for r in rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        a = np.array([r["alpha"] for r in data])
        ls = "-" if formulation == "unscaled" else "--"
        for bk, label, color in bk_info:
            mu = np.array([r[f"{bk}_mean"] for r in data])
            lbl = f"{label}" if formulation == "unscaled" else None
            ax.plot(a, mu, ls, color=color, lw=1.8, label=lbl)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\|J_{ij}\|_F$")
    ax.set_title("(a) Block Frobenius norms\n(solid = unscaled, dashed = scaled)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # ---- (b) sigma_max vs alpha -------------------------------------------
    ax = axes[1]
    expected_b = {"unscaled": -1.0, "scaled_raw": 0.0}
    for formulation in formulations:
        marker, color = sty[formulation]
        data = sorted(
            [r for r in rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        log_a = np.array([np.log10(r["alpha"]) for r in data])
        mu = np.array([r["log10_sigma_max_mean"] for r in data])
        sd = np.array([r["log10_sigma_max_std"]  for r in data])
        slope, intercept = np.polyfit(log_a, mu, 1)
        exp = expected_b[formulation]
        ax.errorbar(log_a, mu, yerr=sd, fmt=f"{marker}-", color=color,
                    capsize=4, lw=1.5,
                    label=f"{formulation} (fit {slope:+.2f}, expect {exp:+.1f})")
        la = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la, intercept + slope * la, color=color, lw=1, ls=":")

    ax.set_xlabel(r"$\log_{10}\;\alpha$")
    ax.set_ylabel(r"$\log_{10}\;\sigma_{\max}(J)$")
    ax.set_title(r"(b) Largest singular value $\sigma_{\max}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (c) Gradient ratio rho -------------------------------------------
    ax = axes[2]
    expected_c = {"unscaled": -1.0, "scaled_raw": 0.0}
    for formulation in formulations:
        marker, color = sty[formulation]
        data = sorted(
            [r for r in rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        log_a = np.array([np.log10(r["alpha"]) for r in data])
        mu = np.array([r["log10_rho_mean"] for r in data])
        sd = np.array([r["log10_rho_std"]  for r in data])
        slope, intercept = np.polyfit(log_a, mu, 1)
        exp = expected_c[formulation]
        ax.errorbar(log_a, mu, yerr=sd, fmt=f"{marker}-", color=color,
                    capsize=4, lw=1.5,
                    label=f"{formulation} (fit {slope:+.2f}, expect {exp:+.1f})")
        la = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la, intercept + slope * la, color=color, lw=1, ls=":")

    ax.set_xlabel(r"$\log_{10}\;\alpha$")
    ax.set_ylabel(
        r"$\log_{10}\;\rho\;$"
        r"$=\|\nabla L_{\mathrm{eq}_1}\|/\|\nabla L_{\mathrm{eq}_2}\|$")
    ax.set_title(r"(c) Gradient ratio $\rho$ at init")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment 1 — Jacobian structure: "
        "unscaled (1.4) vs scaled (1.5),  "
        "4×50 tanh net, at initialisation",
        fontsize=12,
    )
    plt.tight_layout()
    png = out_dir / "exp1_analysis.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()