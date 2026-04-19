"""Experiment 1 — Jacobian conditioning (redesigned).

The paper claims that the scaling (1.5) balances the Jacobian block structure
relative to (1.4).  We verify this in three complementary ways:

Figure 1a — Block Frobenius norms
    The Jacobian J has a natural 2×2 block structure:
        J = [ J_11  J_12 ]     rows: equation (eq1 / eq2)
            [ J_21  J_22 ]     cols: sub-network (y_net / p_net)
    Plot ||J_ij||_F vs alpha for both formulations.
    For (1.4):   J_12 ~ alpha^{-1}, others O(1).
    For (1.5):   J_11, J_22 ~ alpha^{1/2}, J_12, J_21 ~ O(1).

Figure 1b — Block imbalance ratio
    rho_block = max ||J_ij||_F / min ||J_ij||_F
    (1.4): slope -1  (rho ~ alpha^{-1})
    (1.5): slope -1/2 (rho ~ alpha^{-1/2})
    This is robust, works with any network, and does not require sigma_min.

Figure 1c — Full condition number kappa(J) with a tiny network
    depth=1, width=5, P~42 parameters, N_r=100 (J is 200x42, over-determined).
    sigma_min is meaningful because base kappa ~ 10^{1-3}.
    (1.4): slope -1
    (1.5): slope -0.5

Output:
    results/exp1_block_norms.csv     (Figures 1a, 1b)
    results/exp1_kappa_tiny.csv      (Figure 1c)
    results/exp1_conditioning.png    (all three panels)
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
)

# ---- Constants ------------------------------------------------------------

ALPHAS = (1.0, 1e-2, 1e-4, 1e-6)
SEEDS  = (0, 1, 2)

# Block-norm experiment: uses any network size (default 4x50 from the paper).
BLOCK_WIDTH = 50
BLOCK_DEPTH = 4
BLOCK_NR    = 500       # only needs enough points for stable Frobenius norms

# Tiny-net kappa experiment: must be over-determined (2*N_r > P).
# depth=1, width=5: P = 2 * [(2*5+5)+(5*1+1)] = 2*21 = 42
# N_r=100: J is 200 x 42.  ✓
TINY_WIDTH = 5
TINY_DEPTH = 1
TINY_NR    = 100


# ---------------------------------------------------------------------------
# Jacobian assembly (reverse-mode, row-by-row, float64 on CPU)
# ---------------------------------------------------------------------------
def assemble_jacobian(net_y, net_p, x, mms, formulation) -> torch.Tensor:
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
# Split J into 2x2 blocks by (equation, sub-network)
# ---------------------------------------------------------------------------
def split_blocks(J: torch.Tensor, N_r: int, P_y: int):
    """Split J (2*N_r, P_y+P_p) into four blocks.

    Rows:  first N_r = eq1,  last N_r = eq2
    Cols:  first P_y = y_net params,  rest = p_net params
    """
    J11 = J[:N_r,  :P_y]     # eq1, y_net
    J12 = J[:N_r,  P_y:]     # eq1, p_net
    J21 = J[N_r:,  :P_y]     # eq2, y_net
    J22 = J[N_r:,  P_y:]     # eq2, p_net
    return J11, J12, J21, J22


# ---------------------------------------------------------------------------
# Clone to CPU float64
# ---------------------------------------------------------------------------
def _to_f64(net_y, net_p, x):
    ny = copy.deepcopy(net_y).cpu().to(torch.float64)
    np_ = copy.deepcopy(net_p).cpu().to(torch.float64)
    xx = x.detach().cpu().to(torch.float64)
    return ny, np_, xx


# ---------------------------------------------------------------------------
# Part 1: Block Frobenius norms  (any network size)
# ---------------------------------------------------------------------------
def block_norms_one_point(alpha, formulation, seed, device):
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=BLOCK_WIDTH, depth=BLOCK_DEPTH
    )
    x = sample_interior(BLOCK_NR, seed=seed + 1000, device=device)

    ny64, np64, x64 = _to_f64(net_y, net_p, x)
    J = assemble_jacobian(ny64, np64, x64, mms, formulation)

    P_y = sum(p.numel() for p in ny64.parameters())
    J11, J12, J21, J22 = split_blocks(J, BLOCK_NR, P_y)

    norms = {
        "J11": torch.norm(J11).item(),   # eq1, y_net
        "J12": torch.norm(J12).item(),   # eq1, p_net
        "J21": torch.norm(J21).item(),   # eq2, y_net
        "J22": torch.norm(J22).item(),   # eq2, p_net
    }
    norms["ratio"] = max(norms.values()) / max(min(norms.values()), 1e-300)
    return norms


def run_block_norms(device):
    """Sweep alpha × formulation × seed; return list of row dicts."""
    rows = []
    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    for formulation in formulations:
        for alpha in ALPHAS:
            seed_data = {k: [] for k in ("J11", "J12", "J21", "J22", "ratio")}
            for seed in SEEDS:
                ts = time.time()
                n = block_norms_one_point(alpha, formulation, seed, device)
                for k in seed_data:
                    seed_data[k].append(n[k])
                print(
                    f"  [block] {formulation:<10s} a={alpha:.0e} s={seed}  "
                    f"J11={n['J11']:.2e} J12={n['J12']:.2e} "
                    f"J21={n['J21']:.2e} J22={n['J22']:.2e}  "
                    f"ratio={n['ratio']:.2e}  ({time.time()-ts:.1f}s)"
                )
            row = {"formulation": formulation, "alpha": alpha}
            for k in seed_data:
                arr = np.array(seed_data[k])
                row[f"{k}_mean"] = float(np.mean(arr))
                row[f"{k}_std"]  = float(np.std(arr))
            # log10 of ratio for slope fitting
            row["log10_ratio_mean"] = float(np.mean(np.log10(seed_data["ratio"])))
            row["log10_ratio_std"]  = float(np.std(np.log10(seed_data["ratio"])))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Part 2: Full kappa(J) with tiny network  (over-determined)
# ---------------------------------------------------------------------------
def kappa_one_point(alpha, formulation, seed, device):
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(
        device, seed=seed, width=TINY_WIDTH, depth=TINY_DEPTH
    )
    x = sample_interior(TINY_NR, seed=seed + 1000, device=device)

    ny64, np64, x64 = _to_f64(net_y, net_p, x)
    J = assemble_jacobian(ny64, np64, x64, mms, formulation)
    m, n = J.shape

    # Over-determined: use J^T J (n x n); all n eigenvalues are genuine SVs^2
    assert m >= n, f"System must be over-determined: {m} rows < {n} cols"
    A = J.numpy().T @ J.numpy()
    eigs = np.linalg.eigvalsh(A)
    eigs = np.clip(eigs, 0.0, None)
    s_max = float(np.sqrt(eigs[-1]))
    s_min = float(np.sqrt(max(eigs[0], 1e-300)))

    log10_kJ   = math.log10(s_max) - math.log10(s_min)
    log10_kJTJ = 2.0 * log10_kJ
    return {
        "s_max": s_max, "s_min": s_min,
        "log10_kJ": log10_kJ, "log10_kJTJ": log10_kJTJ,
        "shape": (m, n),
    }


def run_kappa_tiny(device):
    rows = []
    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    for formulation in formulations:
        for alpha in ALPHAS:
            kJ_list, kJTJ_list = [], []
            for seed in SEEDS:
                ts = time.time()
                r = kappa_one_point(alpha, formulation, seed, device)
                kJ_list.append(r["log10_kJ"])
                kJTJ_list.append(r["log10_kJTJ"])
                print(
                    f"  [kappa] {formulation:<10s} a={alpha:.0e} s={seed}  "
                    f"J={r['shape']}  "
                    f"s_max={r['s_max']:.2e} s_min={r['s_min']:.2e}  "
                    f"log10 k(J)={r['log10_kJ']:.2f}  "
                    f"({time.time()-ts:.1f}s)"
                )
            rows.append({
                "formulation": formulation,
                "alpha": alpha,
                "log10_kJ_mean":   float(np.mean(kJ_list)),
                "log10_kJ_std":    float(np.std(kJ_list)),
                "log10_kJTJ_mean": float(np.mean(kJTJ_list)),
                "log10_kJTJ_std":  float(np.std(kJTJ_list)),
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(block_rows, kappa_rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    styles = {
        "unscaled":   ("o", "tomato"),
        "scaled_raw": ("s", "steelblue"),
    }
    block_keys = ["J11", "J12", "J21", "J22"]
    block_labels = {
        "J11": r"$\|J_{11}\|$ (eq1, $y$-net)",
        "J12": r"$\|J_{12}\|$ (eq1, $p$-net)",
        "J21": r"$\|J_{21}\|$ (eq2, $y$-net)",
        "J22": r"$\|J_{22}\|$ (eq2, $p$-net)",
    }
    block_colors = {"J11": "C0", "J12": "C1", "J21": "C2", "J22": "C3"}

    # ---- Panel (a): Block norms vs alpha, one sub-plot per formulation idea
    # Actually, put both formulations in one panel with solid vs dashed
    ax = axes[0]
    for formulation in ("unscaled", "scaled_raw"):
        data = sorted(
            [r for r in block_rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        a = np.array([r["alpha"] for r in data])
        ls = "-" if formulation == "unscaled" else "--"
        for bk in block_keys:
            mu = np.array([r[f"{bk}_mean"] for r in data])
            ax.plot(a, mu, f"{ls}", color=block_colors[bk], lw=1.5,
                    label=f"{formulation} {block_labels[bk]}" if formulation == "unscaled" else None)
            if formulation == "scaled_raw":
                ax.plot(a, mu, f"{ls}", color=block_colors[bk], lw=1.5, alpha=0.6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\|J_{ij}\|_F$")
    ax.set_title("(a) Block Frobenius norms\n(solid=unscaled, dashed=scaled)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    # ---- Panel (b): Block imbalance ratio
    ax = axes[1]
    expected_slopes = {"unscaled": -1.0, "scaled_raw": -0.5}
    for formulation in ("unscaled", "scaled_raw"):
        marker, color = styles[formulation]
        data = sorted(
            [r for r in block_rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        log_a = np.array([np.log10(r["alpha"]) for r in data])
        mu    = np.array([r["log10_ratio_mean"] for r in data])
        sd    = np.array([r["log10_ratio_std"]  for r in data])
        slope, intercept = np.polyfit(log_a, mu, 1)
        exp = expected_slopes[formulation]
        ax.errorbar(log_a, mu, yerr=sd, fmt=f"{marker}-", color=color,
                    capsize=4, lw=1.5,
                    label=f"{formulation} (fit {slope:+.2f}, expect {exp:+.1f})")
        la_fit = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la_fit, intercept + slope * la_fit, color=color, lw=1, ls=":")

    ax.set_xlabel(r"$\log_{10}\;\alpha$")
    ax.set_ylabel(r"$\log_{10}\;\rho_{\mathrm{block}}$")
    ax.set_title(r"(b) Block imbalance $\max\|J_{ij}\|/\min\|J_{ij}\|$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ---- Panel (c): kappa(J) from tiny network
    ax = axes[2]
    expected_slopes_k = {"unscaled": -1.0, "scaled_raw": -0.5}
    for formulation in ("unscaled", "scaled_raw"):
        marker, color = styles[formulation]
        data = sorted(
            [r for r in kappa_rows if r["formulation"] == formulation],
            key=lambda r: r["alpha"],
        )
        log_a = np.array([np.log10(r["alpha"]) for r in data])
        mu    = np.array([r["log10_kJ_mean"] for r in data])
        sd    = np.array([r["log10_kJ_std"]  for r in data])
        slope, intercept = np.polyfit(log_a, mu, 1)
        exp = expected_slopes_k[formulation]
        ax.errorbar(log_a, mu, yerr=sd, fmt=f"{marker}-", color=color,
                    capsize=4, lw=1.5,
                    label=f"{formulation} (fit {slope:+.2f}, expect {exp:+.1f})")
        la_fit = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la_fit, intercept + slope * la_fit, color=color, lw=1, ls=":")

    ax.set_xlabel(r"$\log_{10}\;\alpha$")
    ax.set_ylabel(r"$\log_{10}\;\kappa(J)$")
    ax.set_title(r"(c) $\kappa(J)$ — tiny net (depth 1, width 5)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Experiment 1 — Jacobian structure: unscaled (1.4) vs scaled (1.5), at initialisation",
        fontsize=12,
    )
    plt.tight_layout()
    png = out_dir / "exp1_conditioning.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("Part 1: Block Frobenius norms (full-size network)")
    print("=" * 60)
    block_rows = run_block_norms(device)

    # Save CSV
    csv1 = out_dir / "exp1_block_norms.csv"
    with csv1.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(block_rows[0].keys()))
        w.writeheader()
        w.writerows(block_rows)
    print(f"Wrote {csv1}")

    print()
    print("=" * 60)
    print("Part 2: kappa(J) with tiny network")
    print("=" * 60)
    kappa_rows = run_kappa_tiny(device)

    csv2 = out_dir / "exp1_kappa_tiny.csv"
    with csv2.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kappa_rows[0].keys()))
        w.writeheader()
        w.writerows(kappa_rows)
    print(f"Wrote {csv2}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # Plot
    make_plots(block_rows, kappa_rows, out_dir)


if __name__ == "__main__":
    main()


