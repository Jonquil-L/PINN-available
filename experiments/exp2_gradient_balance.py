"""Experiment 2 — Gradient imbalance across the two KKT equations.

Claim: For the unscaled system (1.4), the ratio
    rho(phi) = ||grad_phi L_eq1|| / ||grad_phi L_eq2||
is off by roughly alpha^-2; for the scaled system (1.5) it is O(1).

Procedure: train each (alpha, formulation) with Adam(lr=1e-3) for 10000
iterations, recording rho every 100 iterations. Additionally, save per-layer
histograms of grad L_eq1 and grad L_eq2 at iteration 10000 for alpha=1e-4.

Output:
    results/exp2_rho_{formulation}_{alpha}.npy   (per-run rho curves)
    results/exp2_rho.png
    results/exp2_layer_hist_{formulation}.png    (only for alpha=1e-4)
"""
from __future__ import annotations

import os
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
        flat_grad,
        loss_terms,
        param_list,
        pick_device,
        sample_interior,
    )
except ImportError:
    # Allow direct execution: python experiments/exp2_gradient_balance.py
    from common import (  # type: ignore[no-redef]
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

ITERS = 5000
RECORD_EVERY = 100
PRINT_EVERY = 1_000
SEEDS = (0, 1, 2)


def run_one(alpha: float, formulation: Formulation, seed: int, device, collect_layer_hists: bool):
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    params = param_list(net_y, net_p)
    opt = torch.optim.Adam(params, lr=1e-3)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    rho_hist: list[float] = []
    layer_grads_eq1: dict[str, np.ndarray] = {}
    layer_grads_eq2: dict[str, np.ndarray] = {}

    named_params = (
        [(f"y_{n}", p) for n, p in net_y.named_parameters()]
        + [(f"p_{n}", p) for n, p in net_p.named_parameters()]
    )

    for it in range(ITERS + 1):
        opt.zero_grad()
        l1, l2 = loss_terms(net_y, net_p, x, mms, formulation)
        loss = l1 + l2

        if it % RECORD_EVERY == 0:
            g1 = flat_grad(l1, params, retain_graph=True)
            g2 = flat_grad(l2, params, retain_graph=True)
            r = (g1.norm() / g2.norm().clamp_min(1e-30)).item()
            rho_hist.append(r)

            if collect_layer_hists and it == ITERS:
                # per-layer grad samples (flat arrays)
                # Recompute per-param grads separately to tag by layer name.
                grads1 = torch.autograd.grad(l1, params, retain_graph=True, allow_unused=True)
                grads2 = torch.autograd.grad(l2, params, retain_graph=True, allow_unused=True)
                for (name, _p), g1p, g2p in zip(named_params, grads1, grads2):
                    if g1p is not None and "weight" in name:
                        layer_grads_eq1[name] = g1p.detach().cpu().numpy().ravel()
                        layer_grads_eq2[name] = g2p.detach().cpu().numpy().ravel()

        if it % PRINT_EVERY == 0:
            print(
                f"[{formulation} a={alpha:.0e} seed={seed}] "
                f"iter {it:5d}/{ITERS} loss={loss.item():.3e}"
            )

        loss.backward()
        opt.step()

    return np.array(rho_hist), layer_grads_eq1, layer_grads_eq2


def main():
    device = pick_device()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    iters_axis = np.arange(0, ITERS + 1, RECORD_EVERY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(ALPHAS_STANDARD)))

    formulations: tuple[Formulation, Formulation] = ("unscaled", "scaled")
    for ax, formulation in zip(axes, formulations):
        for c, alpha in zip(colors, ALPHAS_STANDARD):
            seed_runs = []
            hist_eq1, hist_eq2 = {}, {}
            for seed in SEEDS:
                want_hists = (alpha == 1e-4 and seed == SEEDS[0])
                rho, h1, h2 = run_one(alpha, formulation, seed, device, want_hists)
                seed_runs.append(rho)
                if want_hists:
                    hist_eq1, hist_eq2 = h1, h2
            rho_arr = np.stack(seed_runs, axis=0)
            np.save(out_dir / f"exp2_rho_{formulation}_a{alpha:.0e}.npy", rho_arr)

            median = np.median(rho_arr, axis=0)
            lo, hi = rho_arr.min(axis=0), rho_arr.max(axis=0)
            ax.fill_between(iters_axis, lo, hi, color=c, alpha=0.2)
            ax.plot(iters_axis, median, color=c, label=rf"$\alpha$={alpha:.0e}")

            if hist_eq1:
                # Plot histograms for alpha=1e-4 only.
                fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
                for name, arr in hist_eq1.items():
                    ax2.hist(np.log10(np.abs(arr) + 1e-40), bins=60, alpha=0.5,
                             label=f"eq1 {name}", histtype="step")
                for name, arr in hist_eq2.items():
                    ax2.hist(np.log10(np.abs(arr) + 1e-40), bins=60, alpha=0.5,
                             label=f"eq2 {name}", histtype="step", linestyle="--")
                ax2.set_xlabel(r"$\log_{10}|\nabla_\phi L|$ (per layer)")
                ax2.set_ylabel("count")
                ax2.set_title(f"Per-layer gradient magnitudes — {formulation}, alpha=1e-4, iter {ITERS}")
                ax2.legend(fontsize=7, ncol=2)
                fig2.tight_layout()
                fig2.savefig(out_dir / f"exp2_layer_hist_{formulation}.png", dpi=150)
                plt.close(fig2)

        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        if formulation == "unscaled":
            ax.set_ylabel(r"$\rho(\phi)=\|\nabla L_{\mathrm{eq}_1}\|/\|\nabla L_{\mathrm{eq}_2}\|$")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{formulation} — (1.{'4' if formulation=='unscaled' else '5'})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Experiment 2 — gradient ratio across the two equations")
    plt.tight_layout()
    png = out_dir / "exp2_rho.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
