"""Plot 2 — Largest singular value sigma_max(J) vs alpha.

Reads results/exp0_data.csv and writes results/exp0_fig_b_sigma_max.png.

Theoretical prediction (Theorem 3.4 of the review):
    Unscaled (1.4): sigma_max(J) = Theta(alpha^{-1})  -> slope -1.
    Scaled (1.5):   sigma_max(J) = Theta(1)          -> slope  0.

Median over 10 seeds. Slope fit via weighted least squares in log-log
space; error bars = std of log10 values across seeds.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("results") / "exp0_data.csv"
OUT_PATH = Path("results") / "exp0_fig_b_sigma_max.png"


def _load():
    data = defaultdict(list)
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["formulation"], float(row["alpha"]))
            data[key].append(float(row["sigma_max"]))
    return data


def main():
    data = _load()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    expected = {"unscaled": -1.0, "scaled_raw": 0.0}
    style    = {
        "unscaled":   ("o", "tomato",    "(1.4) unscaled"),
        "scaled_raw": ("s", "steelblue", "(1.5) scaled"),
    }

    for formulation in ("unscaled", "scaled_raw"):
        pts = sorted(
            (alpha, vals)
            for (f, alpha), vals in data.items() if f == formulation
        )
        log_a = np.array([np.log10(a) for a, _ in pts])
        log_mu = np.array([np.mean(np.log10(v))  for _, v in pts])
        log_sd = np.array([np.std (np.log10(v))  for _, v in pts])

        slope, intercept = np.polyfit(log_a, log_mu, 1)
        marker, color, fname = style[formulation]

        ax.errorbar(
            log_a, log_mu, yerr=log_sd,
            fmt=f"{marker}-", color=color, capsize=4, lw=1.8, ms=8,
            label=(f"{fname}   fit slope = {slope:+.2f}   "
                   f"theory = {expected[formulation]:+.1f}"),
        )
        # Dotted fit line.
        la = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la, intercept + slope * la,
                color=color, lw=0.9, ls=":", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}\,\alpha$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}\,\sigma_{\max}(J)$", fontsize=12)
    ax.set_title(
        r"Largest singular value of the PINN Jacobian at initialisation"
        "\n"
        r"error bars = std of $\log_{10}\sigma_{\max}$ across 10 seeds",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()