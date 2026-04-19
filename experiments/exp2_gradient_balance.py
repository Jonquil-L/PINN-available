"""Experiment 2 — Gradient imbalance across the two KKT equations.

Claim: For the unscaled system (1.4), the ratio
    rho(phi) = ||grad_phi L_eq1|| / ||grad_phi L_eq2||
is off by roughly alpha^{-1}; for the scaled system (1.5) it is O(1).

This is a DIAGNOSTIC, so we use the raw residuals of (1.5) -- formulation
"scaled_raw" in common.py -- not the old loss-normalised variant.

Outputs:
    results/exp2_rho_vs_alpha.png        PRIMARY figure: rho at final iter vs alpha
    results/exp2_rho.png                 Supplementary: rho vs iteration curves
    results/exp2_hist_aggregated.png     Aggregated gradient histograms at alpha=1e-4
    results/exp2_rho_{formulation}_a{alpha}.npy
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import (
    ALPHAS_STANDARD,
    Formulation,
    ManufacturedSolution,
    build_networks,
    flat_grad,
    loss_terms,
    param_list,
    pick_device,
    sample_interior,
)

ITERS = 10_000
RECORD_EVERY = 100
SEEDS = (0, 1, 2)


def run_one(alpha: float, formulation: Formulation, seed: int, device,
            collect_agg_hists: bool = False):
    """Train and track rho = ||grad L_eq1|| / ||grad L_eq2|| over iterations.

    If collect_agg_hists is True, also return the concatenated flat gradient
    vectors at the final iteration (aggregated across all params).
    """
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    params = param_list(net_y, net_p)
    opt = torch.optim.Adam(params, lr=1e-3)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    rho_hist: list[float] = []
    agg_g1: np.ndarray | None = None
    agg_g2: np.ndarray | None = None

    for it in range(ITERS + 1):
        opt.zero_grad()
        l1, l2 = loss_terms(net_y, net_p, x, mms, formulation)

        if it % RECORD_EVERY == 0:
            g1 = flat_grad(l1, params, retain_graph=True)
            g2 = flat_grad(l2, params, retain_graph=True)
            r = (g1.norm() / g2.norm().clamp_min(1e-30)).item()
            rho_hist.append(r)

            if collect_agg_hists and it == ITERS:
                agg_g1 = g1.detach().cpu().numpy()
                agg_g2 = g2.detach().cpu().numpy()

        loss = l1 + l2
        loss.backward()
        opt.step()

    return np.array(rho_hist), agg_g1, agg_g2


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    iters_axis = np.arange(0, ITERS + 1, RECORD_EVERY)
    formulations: tuple[Formulation, ...] = ("unscaled", "scaled_raw")
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(ALPHAS_STANDARD)))

    # Collect all results first so we can build multiple figures.
    # rho_data[formulation][alpha] = (rho_arr shape (n_seeds, n_record_pts),)
    rho_data: dict[str, dict[float, np.ndarray]] = {f: {} for f in formulations}
    # Aggregated histograms at alpha=1e-4 for each formulation
    hist_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for formulation in formulations:
        for alpha in ALPHAS_STANDARD:
            seed_runs = []
            for seed in SEEDS:
                want_hists = (alpha == 1e-4 and seed == SEEDS[0])
                rho, g1, g2 = run_one(alpha, formulation, seed, device, want_hists)
                seed_runs.append(rho)
                if g1 is not None:
                    hist_data[formulation] = (g1, g2)
            rho_arr = np.stack(seed_runs, axis=0)
            rho_data[formulation][alpha] = rho_arr
            np.save(out_dir / f"exp2_rho_{formulation}_a{alpha:.0e}.npy", rho_arr)
            final_rho = np.median(rho_arr[:, -1])
            print(f"{formulation:<12s} alpha={alpha:.0e}  rho_final={final_rho:.3e}")

    # ---------------------------------------------------------------
    # PRIMARY FIGURE: rho at final iteration vs alpha (C2)
    # ---------------------------------------------------------------
    fig_primary, ax_p = plt.subplots(figsize=(6, 4.5))
    for formulation, marker, color in (("unscaled", "o", "tomato"), ("scaled_raw", "s", "steelblue")):
        alphas = sorted(rho_data[formulation].keys())
        rho_final = []
        for a in alphas:
            rho_final.append(np.median(rho_data[formulation][a][:, -1]))
        a_arr = np.array(alphas)
        rho_arr = np.array(rho_final)
        slope, intercept = np.polyfit(np.log10(a_arr), np.log10(rho_arr), 1)
        ax_p.loglog(a_arr, rho_arr, f"{marker}-", color=color,
                    label=f"{formulation} (slope {slope:+.2f})")
        a_fit = np.geomspace(a_arr.min(), a_arr.max(), 50)
        ax_p.plot(a_fit, 10 ** (intercept + slope * np.log10(a_fit)),
                  color=color, lw=0.8, ls=":")
    ax_p.axhline(1.0, color="k", lw=0.5, ls=":")
    ax_p.invert_xaxis()
    ax_p.set_xlabel(r"$\alpha$")
    ax_p.set_ylabel(r"$\rho = \|\nabla_\phi L_{\mathrm{eq}_1}\| / \|\nabla_\phi L_{\mathrm{eq}_2}\|$")
    ax_p.set_title(r"$\rho$ at final iteration vs $\alpha$ (expected: unscaled slope $\approx -1$, scaled_raw $\approx 0$)")
    ax_p.grid(True, which="both", alpha=0.3)
    ax_p.legend()
    fig_primary.tight_layout()
    fig_primary.savefig(out_dir / "exp2_rho_vs_alpha.png", dpi=150)
    print(f"Wrote {out_dir / 'exp2_rho_vs_alpha.png'}")
    plt.close(fig_primary)

    # ---------------------------------------------------------------
    # SUPPLEMENTARY: rho vs iteration (keep existing style) (C4)
    # ---------------------------------------------------------------
    fig_rho, axes_rho = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, formulation in zip(axes_rho, formulations):
        for c, alpha in zip(colors, ALPHAS_STANDARD):
            rho_arr = rho_data[formulation][alpha]
            median = np.median(rho_arr, axis=0)
            lo, hi = rho_arr.min(axis=0), rho_arr.max(axis=0)
            ax.fill_between(iters_axis, lo, hi, color=c, alpha=0.2)
            ax.plot(iters_axis, median, color=c, label=rf"$\alpha$={alpha:.0e}")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        if formulation == "unscaled":
            ax.set_ylabel(r"$\rho(\phi)$")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        _label = {"unscaled": "1.4", "scaled_raw": "1.5"}[formulation]
        ax.set_title(f"{formulation} — ({_label})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig_rho.suptitle("Experiment 2 — gradient ratio vs iteration (supplementary)")
    fig_rho.tight_layout()
    fig_rho.savefig(out_dir / "exp2_rho.png", dpi=150)
    print(f"Wrote {out_dir / 'exp2_rho.png'}")
    plt.close(fig_rho)

    # ---------------------------------------------------------------
    # AGGREGATED HISTOGRAM at alpha=1e-4 (C3)
    # Two panels: unscaled | scaled_raw.  Each panel shows two
    # histograms: log10|g| for eq1 (solid) and eq2 (dashed).
    # ---------------------------------------------------------------
    fig_hist, axes_h = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, formulation in zip(axes_h, formulations):
        if formulation not in hist_data:
            continue
        g1, g2 = hist_data[formulation]
        log_g1 = np.log10(np.abs(g1) + 1e-40)
        log_g2 = np.log10(np.abs(g2) + 1e-40)
        ax.hist(log_g1, bins=80, alpha=0.6, label="eq1 (state)", histtype="stepfilled", color="C0")
        ax.hist(log_g2, bins=80, alpha=0.6, label="eq2 (adjoint)", histtype="stepfilled", color="C1")
        ax.set_xlabel(r"$\log_{10}|\nabla_\phi L_i|$")
        ax.set_ylabel("count")
        _label = {"unscaled": "1.4", "scaled_raw": "1.5"}[formulation]
        ax.set_title(f"{formulation} — ({_label}), alpha=1e-4, iter {ITERS}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig_hist.suptitle("Experiment 2 — aggregated gradient magnitudes")
    fig_hist.tight_layout()
    fig_hist.savefig(out_dir / "exp2_hist_aggregated.png", dpi=150)
    print(f"Wrote {out_dir / 'exp2_hist_aggregated.png'}")
    plt.close(fig_hist)


if __name__ == "__main__":
    main()
