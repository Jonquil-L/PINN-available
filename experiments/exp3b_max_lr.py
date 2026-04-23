"""Experiment 3c — Maximum stable learning rate (corrected).

Fixes the two bugs in exp3b:

Bug 1: Conflated stability and convergence. The criterion
    "final loss <= initial/1000 in 5000 steps"
is a CONVERGENCE test, not a STABILITY test. A stable-but-slow trajectory
is still stable. Theory predicts eta_max ~ 2/sigma_max^2 is where the
discrete linearised iteration becomes divergent, i.e. loss grows
geometrically. That is what we should detect.

Bug 2: Short floor lr=1e-8 was too small to ever "pass" the (wrong)
convergence criterion, so bisection never got off the ground and
returned 0 everywhere.

Corrected criterion: a trial is "stable" iff
    (a) loss and all gradients remain finite throughout training,
    (b) final loss <= DIVERGENCE_FACTOR * initial loss (default 10x).

This is the standard edge-of-stability definition used in optimization
theory. A trajectory that drifts slowly upward by 2x is stable; one
that blows up by 100x or NaNs is not.

Other changes from exp3b:
  - ITERS reduced to 1000 (stability decision is geometric, emerges fast).
  - LR_LO_INIT = 1e-10, accepted as trivially stable.
  - Per-trial logging reduced to one line.

Theoretical prediction (Theorem 3.6 of the review):
    eta_max^{(1.4)} = Theta(alpha^2)   (slope +2)
    eta_max^{(1.5)} = Theta(1)         (slope 0)

Output:
    results/exp3c_eta_max.csv
    results/exp3c_eta_max.png
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

try:
    from .common import (
        Formulation,
        ManufacturedSolution,
        build_networks,
        param_list,
        pick_device,
        sample_interior,
        total_loss,
    )
except ImportError:
    from common import (
        Formulation,
        ManufacturedSolution,
        build_networks,
        param_list,
        pick_device,
        sample_interior,
        total_loss,
    )


# ---- Config ---------------------------------------------------------------

ALPHAS            = (1.0, 1e-2, 1e-4, 1e-6, 1e-8)
SEEDS             = tuple(range(10))
STABILITY_K       = 8                   # >=8/10 seeds must pass
ITERS             = 1_000               # enough for divergence to emerge
DIVERGENCE_FACTOR = 10.0                # final <= 10 x initial = stable
BISECT_STEPS      = 12                  # ~3.6 decades of LR resolution
LR_LO_INIT        = 1e-10               # trivially stable floor
LR_HI_INIT        = 1.0                 # try surprisingly-large ceiling

VERBOSE = False                          # set True to log per-trial


# ---- Stability test -------------------------------------------------------

def run_trial(alpha: float, formulation: Formulation, lr: float,
              seed: int, device) -> bool:
    """Stable iff (a) no NaN throughout, (b) final loss <= 10 x initial."""
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    params = param_list(net_y, net_p)
    opt = torch.optim.Adam(params, lr=lr)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    # Initial loss — do NOT use torch.no_grad, the Laplacian inside residuals
    # needs the autograd graph to compute second derivatives.
    l0_tensor = total_loss(net_y, net_p, x, mms, formulation)
    l0 = l0_tensor.detach().item()
    del l0_tensor
    if not math.isfinite(l0):
        return False
    divergence_threshold = DIVERGENCE_FACTOR * max(l0, 1e-30)

    for it in range(ITERS):
        opt.zero_grad()
        loss = total_loss(net_y, net_p, x, mms, formulation)
        loss_val = loss.detach().item()
        if not math.isfinite(loss_val):
            return False
        if loss_val > divergence_threshold:
            return False
        loss.backward()
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        opt.step()

    # Final check — again, no no_grad.
    lf_tensor = total_loss(net_y, net_p, x, mms, formulation)
    lf = lf_tensor.detach().item()
    if not math.isfinite(lf):
        return False
    if lf > divergence_threshold:
        return False
    return True



def prob_stable(alpha: float, formulation: Formulation, lr: float,
                device) -> float:
    n_ok = 0
    for seed in SEEDS:
        ok = run_trial(alpha, formulation, lr, seed, device)
        if VERBOSE:
            print(f"    seed={seed} lr={lr:.2e} -> {'OK' if ok else 'FAIL'}")
        if ok:
            n_ok += 1
    return n_ok / len(SEEDS)


# ---- Bisection in log-space ----------------------------------------------

def bisect_eta_max(alpha: float, formulation: Formulation, device) -> float:
    thresh = STABILITY_K / len(SEEDS)

    # Floor check: lr=1e-10 should always be stable on any finite problem.
    p_lo = prob_stable(alpha, formulation, LR_LO_INIT, device)
    if p_lo < thresh:
        # Something is very wrong — likely the loss itself is non-finite at init.
        print(f"  WARN: floor lr={LR_LO_INIT:.0e} is unstable "
              f"(p={p_lo:.2f}). Check initial loss.")
        return 0.0

    # Ceiling check.
    p_hi = prob_stable(alpha, formulation, LR_HI_INIT, device)
    if p_hi >= thresh:
        # Even lr=1 is stable. Theory says this can happen for scaled at
        # large alpha. Try extending upward.
        lr = LR_HI_INIT
        while lr < 1e2:
            lr *= 3.0
            if prob_stable(alpha, formulation, lr, device) < thresh:
                log_hi = math.log10(lr)
                log_lo = math.log10(lr / 3.0)
                break
        else:
            return LR_HI_INIT * 100.0  # saturation: report as >= 100
    else:
        log_lo = math.log10(LR_LO_INIT)
        log_hi = math.log10(LR_HI_INIT)

    for _ in range(BISECT_STEPS):
        log_mid = 0.5 * (log_lo + log_hi)
        lr_mid = 10 ** log_mid
        p = prob_stable(alpha, formulation, lr_mid, device)
        if p >= thresh:
            log_lo = log_mid
        else:
            log_hi = log_mid

    return 10 ** log_lo


# ---- Main sweep -----------------------------------------------------------

def main():
    device = pick_device()
    print(f"Device: {device}")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for formulation in ("unscaled", "scaled_raw"):
        for alpha in ALPHAS:
            t_cell = time.time()
            eta_max = bisect_eta_max(alpha, formulation, device)
            dt = time.time() - t_cell
            print(f"{formulation:<12s} alpha={alpha:.0e}  "
                  f"eta_max={eta_max:.3e}  ({dt:.1f}s)")
            rows.append({
                "formulation": formulation,
                "alpha": alpha,
                "eta_max": eta_max,
            })
    print(f"Total: {time.time() - t0:.1f}s")

    csv_path = out_dir / "exp3c_eta_max.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ---- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    expected = {"unscaled": 2.0, "scaled_raw": 0.0}
    style = {"unscaled": ("o", "tomato"),
             "scaled_raw": ("s", "steelblue")}
    for formulation in ("unscaled", "scaled_raw"):
        data = sorted([r for r in rows if r["formulation"] == formulation],
                      key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in data])
        e = np.array([r["eta_max"] for r in data])
        mask = e > 0
        marker, color = style[formulation]
        if mask.sum() >= 2:
            m, b = np.polyfit(np.log10(a[mask]), np.log10(e[mask]), 1)
            label = (f"{formulation} (fit slope {m:+.2f}, "
                     f"expect {expected[formulation]:+.1f})")
            a_fit = np.geomspace(a[mask].min(), a[mask].max(), 50)
            ax.plot(a_fit, 10 ** (b + m * np.log10(a_fit)),
                    color=color, lw=0.8, ls=":")
        else:
            label = formulation
        ax.loglog(a, np.where(e > 0, e, np.nan),
                  f"{marker}-", color=color, lw=1.5, ms=8, label=label)

    ax.invert_xaxis()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\eta_{\max}$ (largest stable LR, $\geq 8/10$ seeds)")
    ax.set_title(
        r"Experiment 3c — max stable learning rate"
        "\n"
        r"stability = loss stays within $10\times$ initial over "
        f"{ITERS} iters; {len(SEEDS)} seeds/cell"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    png = out_dir / "exp3c_eta_max.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()


