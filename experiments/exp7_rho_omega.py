"""Experiment — Gradient ratio rho vs loss-weight omega.

This is the principle-level diagnostic that explains the phenomenon-level
omega-robustness results in the paper's Section 4 omega-sensitivity tables.

Fix alpha at the pathological value alpha = 1e-4.
Sweep omega over a wide range.
For each omega, measure

    rho_omega(phi) = ||grad_phi (omega * L_1)|| / ||grad_phi L_2||
                   = omega * ||grad_phi L_1|| / ||grad_phi L_2||

at initialisation, averaged over multiple seeds. Report both

    log10 rho_omega              -- signed imbalance (eq1 vs eq2)
    |log10 rho_omega|            -- distance from balance rho = 1

as functions of log10 omega.

Claim to support:
    Under (1.4), |log10 rho| stays large (>> 0) across the whole omega
    sweep because the baseline alpha^{-1} imbalance dominates any reasonable
    omega. No choice of omega in [1e-2, 1e+2] reaches balance.

    Under (1.5), |log10 rho| is near 0 at omega = 1 and stays modest for
    a wide omega band. This is the principle-level explanation of the
    observed omega-robustness of the scaled system.

Outputs:
    results/exp_rho_vs_omega.csv
    results/exp_rho_vs_omega.png

Usage:
    python -m experiments.exp_rho_vs_omega
    # or
    python experiments/exp_rho_vs_omega.py
"""
from __future__ import annotations

import csv
import os
import platform
import time
from pathlib import Path

# Some Windows Python stacks (e.g., mixed conda/pip MKL/OpenMP deps)
# can load duplicate OpenMP runtimes and abort at plotting time.
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
        build_networks,
        flat_grad,
        loss_terms,
        param_list,
        pick_device,
        sample_interior,
    )
except ImportError:
    # Allow running this file directly as a script
    from common import (
        Formulation,
        ManufacturedSolution,
        build_networks,
        flat_grad,
        loss_terms,
        param_list,
        pick_device,
        sample_interior,
    )


# ---- Fixed experiment parameters ------------------------------------------

ALPHA_FIXED = 1e-4                                # pathological regime
OMEGAS = (1e-2, 1e-1, 1.0, 1e+1, 1e+2)            # matches paper's sweep
SEEDS = tuple(range(10))                          # 10 seeds, sufficient for init stats
N_R = 2500                                        # interior collocation size
NET_WIDTH = 50
NET_DEPTH = 4


# ---------------------------------------------------------------------------
# Core measurement: rho for a single (formulation, seed, omega) triple
# ---------------------------------------------------------------------------
def measure_rho(
    formulation: Formulation, seed: int, omega: float, device
) -> tuple[float, float, float]:
    """Return (rho_omega, g1_norm, g2_norm) at initialisation.

    rho_omega := ||grad_phi (omega * L_1)|| / ||grad_phi L_2||
              =  omega * ||grad_phi L_1|| / ||grad_phi L_2||

    We compute both unweighted norms g1_norm, g2_norm (for logging / CSV)
    and derive rho_omega from them. The network is freshly initialised
    per seed; no training is performed.
    """
    torch.manual_seed(seed)
    mms = ManufacturedSolution(ALPHA_FIXED)
    net_y, net_p = build_networks(
        device, seed=seed, width=NET_WIDTH, depth=NET_DEPTH
    )
    params = param_list(net_y, net_p)
    x = sample_interior(N_R, seed=seed + 1000, device=device)

    l1, l2 = loss_terms(net_y, net_p, x, mms, formulation)
    g1 = flat_grad(l1, params, retain_graph=True)
    g2 = flat_grad(l2, params, retain_graph=False)

    g1_norm = g1.norm().clamp_min(1e-30).item()
    g2_norm = g2.norm().clamp_min(1e-30).item()
    rho_omega = omega * g1_norm / g2_norm
    return rho_omega, g1_norm, g2_norm


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    rows: list[dict] = []
    t0 = time.time()

    for formulation in formulations:
        for omega in OMEGAS:
            rho_list: list[float] = []
            g1_list: list[float] = []
            g2_list: list[float] = []
            for seed in SEEDS:
                ts = time.time()
                rho, g1, g2 = measure_rho(formulation, seed, omega, device)
                rho_list.append(rho)
                g1_list.append(g1)
                g2_list.append(g2)
                print(
                    f"{formulation:<10s} omega={omega:.0e}  seed={seed}  "
                    f"||g1||={g1:.3e}  ||g2||={g2:.3e}  "
                    f"rho={rho:.3e}  ({time.time()-ts:.1f}s)"
                )

            log_rho = np.log10(np.array(rho_list))
            rows.append(
                {
                    "formulation": formulation,
                    "omega": omega,
                    "n_seeds": len(SEEDS),
                    "log10_rho_mean": float(np.mean(log_rho)),
                    "log10_rho_std": float(np.std(log_rho)),
                    "log10_rho_median": float(np.median(log_rho)),
                    "abs_log10_rho_mean": float(np.mean(np.abs(log_rho))),
                    "abs_log10_rho_std": float(np.std(np.abs(log_rho))),
                    "g1_mean": float(np.mean(g1_list)),
                    "g2_mean": float(np.mean(g2_list)),
                }
            )

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # ---- CSV ---------------------------------------------------------------
    csv_path = out_dir / "exp_rho_vs_omega.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sty = {"unscaled": ("o", "tomato"), "scaled_raw": ("s", "steelblue")}
    log_w_annot_x = float(np.max(np.log10(np.array(OMEGAS, dtype=float))))

    # Panel (a): signed log10 rho vs log10 omega
    ax = axes[0]
    for formulation in formulations:
        marker, color = sty[formulation]
        data = sorted(
            [r for r in rows if r["formulation"] == formulation],
            key=lambda r: r["omega"],
        )
        log_w = np.array([np.log10(r["omega"]) for r in data])
        mu = np.array([r["log10_rho_mean"] for r in data])
        sd = np.array([r["log10_rho_std"] for r in data])
        slope, intercept = np.polyfit(log_w, mu, 1)
        label = (
            f"{formulation} "
            f"(slope {slope:+.2f}, expect +1.00)"
        )
        ax.errorbar(
            log_w, mu, yerr=sd,
            fmt=f"{marker}-", color=color, capsize=4, lw=1.5,
            label=label,
        )
        lw = np.linspace(log_w.min(), log_w.max(), 50)
        ax.plot(lw, intercept + slope * lw, color=color, lw=1, ls=":")

    ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.7)
    ax.text(
        log_w_annot_x, 0.05, r"$\rho = 1$ (balanced)",
        color="k", fontsize=9, ha="right", va="bottom", alpha=0.7,
    )
    ax.set_xlabel(r"$\log_{10}\,\omega$")
    ax.set_ylabel(r"$\log_{10}\,\rho_\omega = "
                  r"\log_{10}(\omega\,\|\nabla L_{\mathrm{eq}_1}\|"
                  r"/\|\nabla L_{\mathrm{eq}_2}\|)$")
    ax.set_title(
        r"(a) Signed gradient imbalance at $\alpha=10^{-4}$"
        "\n(theoretical slope $+1$ for both; "
        r"intercept captures $\alpha$-pathology)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    # Panel (b): distance from balance |log10 rho| vs log10 omega
    ax = axes[1]
    for formulation in formulations:
        marker, color = sty[formulation]
        data = sorted(
            [r for r in rows if r["formulation"] == formulation],
            key=lambda r: r["omega"],
        )
        log_w = np.array([np.log10(r["omega"]) for r in data])
        mu = np.array([r["abs_log10_rho_mean"] for r in data])
        sd = np.array([r["abs_log10_rho_std"] for r in data])
        ax.errorbar(
            log_w, mu, yerr=sd,
            fmt=f"{marker}-", color=color, capsize=4, lw=1.5,
            label=formulation,
        )

    ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.7)
    ax.text(
        log_w_annot_x, 0.05, r"$|\log\rho|=0$ (balanced)",
        color="k", fontsize=9, ha="right", va="bottom", alpha=0.7,
    )
    ax.set_xlabel(r"$\log_{10}\,\omega$")
    ax.set_ylabel(r"$|\log_{10}\,\rho_\omega|$  (distance from balance)")
    ax.set_title(
        r"(b) Distance from balance at $\alpha=10^{-4}$"
        "\n(lower = better conditioned for training)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        r"Gradient ratio $\rho$ vs loss weight $\omega$ at $\alpha=10^{-4}$"
        "  (principle-level explanation of $\\omega$-robustness)",
        fontsize=12,
    )
    plt.tight_layout()
    png = out_dir / "exp_rho_vs_omega.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()


