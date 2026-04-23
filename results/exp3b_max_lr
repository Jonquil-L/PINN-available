"""Experiment 3b — Maximum stable learning rate (rewritten).

Replaces exp3_stable_lr.py. Three changes that matter:

1. Stability criterion is stricter and deterministic:
   (i)  no NaN/Inf in loss or gradients throughout training,
   (ii) final loss < initial loss / 10^3 (three-order reduction required),
   (iii) gradient norm never exceeds 10^3 x its running median.

2. LR found by bisection in log-space between a known-unstable lr_hi and a
   known-stable lr_lo, to machine precision (10 bisection steps).

3. Multi-seed: 10 seeds per (alpha, formulation, lr). A given lr is declared
   "stable" at a given alpha iff >= 8/10 seeds pass the criterion. Reported
   eta_max is the largest lr passing this threshold.

Theoretical prediction (§3.6 of review, from sigma_max scaling):
    eta_max^{(1.4)} = Theta(alpha^2)   (slope +2)
    eta_max^{(1.5)} = Theta(1)          (slope 0)

Output:
    results/exp3b_eta_max.csv
    results/exp3b_eta_max.png
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import (
    Formulation,
    ManufacturedSolution,
    build_networks,
    param_list,
    pick_device,
    sample_interior,
    total_loss,
)


# ---- Sweep config ---------------------------------------------------------

ALPHAS        = (1.0, 1e-2, 1e-4, 1e-6, 1e-8)
SEEDS         = tuple(range(10))        # 10 seeds per cell
STABILITY_K   = 8                        # need >=8/10 to call stable
ITERS         = 5_000                    # each trial
BISECT_STEPS  = 10                       # ~3 decades of LR resolution
LR_HI_INIT    = 1e-1                     # known-unstable upper bound
LR_LO_INIT    = 1e-8                     # known-stable lower bound
LOSS_REDUCTION_REQUIRED = 1e-3          # final loss <= initial loss * this


# ---- Stability test (one seed, one lr, one alpha) -------------------------

def run_trial(alpha: float, formulation: Formulation, lr: float,
              seed: int, device) -> bool:
    """Return True if this (alpha, formulation, lr, seed) trial is 'stable'
    per the three-part criterion in the module docstring."""
    torch.manual_seed(seed)
    mms = ManufacturedSolution(alpha)
    net_y, net_p = build_networks(device, seed=seed)
    params = param_list(net_y, net_p)
    opt = torch.optim.Adam(params, lr=lr)
    x = sample_interior(2500, seed=seed + 1000, device=device)

    print(
        f"  trial start: {formulation:<10s} alpha={alpha:.0e} lr={lr:.1e} seed={seed}",
        flush=True,
    )

    # total_loss uses PDE residuals with input derivatives, so do not disable autograd.
    l0 = total_loss(net_y, net_p, x, mms, formulation).item()
    if not math.isfinite(l0) or l0 == 0.0:
        return False

    grad_norms: list[float] = []

    for it in range(ITERS):
        opt.zero_grad()
        loss = total_loss(net_y, net_p, x, mms, formulation)
        if not torch.isfinite(loss):
            print(
                f"    trial abort: non-finite loss at iter={it} seed={seed}",
                flush=True,
            )
            return False
        loss.backward()

        # Collect gradient norm; reject on NaN/Inf.
        g2 = 0.0
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
            g2 += float(p.grad.detach().pow(2).sum().item())
        gnorm = math.sqrt(g2)
        grad_norms.append(gnorm)

        # Spike-detection: after warmup, demand grad norm does not exceed
        # 1000 x its running median for the past 100 iterations.
        if it >= 200:
            recent = np.array(grad_norms[-100:])
            med = float(np.median(recent))
            if med > 0 and gnorm > 1e3 * med:
                print(
                    f"    trial abort: grad spike at iter={it} seed={seed}",
                    flush=True,
                )
                return False

        if it % 500 == 0:
            print(
                f"    iter={it:4d}/{ITERS} seed={seed} loss={float(loss):.3e} gnorm={gnorm:.3e}",
                flush=True,
            )

        opt.step()

    # Same reason as l0: this loss build requires autograd-enabled ops.
    lf = total_loss(net_y, net_p, x, mms, formulation).item()
    if not math.isfinite(lf):
        print(f"    trial abort: non-finite final loss seed={seed}", flush=True)
        return False
    if lf > l0 * LOSS_REDUCTION_REQUIRED:
        print(
            f"    trial fail: final loss {lf:.3e} not reduced enough from {l0:.3e} seed={seed}",
            flush=True,
        )
        return False
    print(f"    trial pass: seed={seed} final_loss={lf:.3e}", flush=True)
    return True


def prob_stable(alpha: float, formulation: Formulation, lr: float,
                device) -> float:
    """Fraction of seeds for which this lr is stable."""
    n_ok = 0
    for seed in SEEDS:
        if run_trial(alpha, formulation, lr, seed, device):
            n_ok += 1
    return n_ok / len(SEEDS)


# ---- Bisection on log(lr) -------------------------------------------------

def bisect_eta_max(alpha: float, formulation: Formulation, device) -> float:
    """Find the largest lr with stability probability >= STABILITY_K/len(SEEDS)
    by bisection in log-space."""
    # Quick check: is the floor stable? If not, there's no stable lr.
    if prob_stable(alpha, formulation, LR_LO_INIT, device) < STABILITY_K / len(SEEDS):
        return 0.0
    # Quick check: is the ceiling stable? If so, the truth is >= ceiling.
    if prob_stable(alpha, formulation, LR_HI_INIT, device) >= STABILITY_K / len(SEEDS):
        return LR_HI_INIT

    log_lo, log_hi = math.log10(LR_LO_INIT), math.log10(LR_HI_INIT)
    for _ in range(BISECT_STEPS):
        log_mid = 0.5 * (log_lo + log_hi)
        lr_mid = 10 ** log_mid
        p = prob_stable(alpha, formulation, lr_mid, device)
        if p >= STABILITY_K / len(SEEDS):
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

    csv_path = out_dir / "exp3b_eta_max.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ---- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
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
            # Plot fit line
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
        r"Experiment 3b — max stable learning rate"
        "\n"
        r"(strict criterion, bisection, 10 seeds; expect slope $+2$ / $0$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    png = out_dir / "exp3b_eta_max.png"
    plt.savefig(png, dpi=150)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()


