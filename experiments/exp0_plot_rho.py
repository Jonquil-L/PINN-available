"""Plot 3 — Gradient ratio rho at initialisation vs alpha.

Reads results/exp0_data.csv and writes results/exp0_fig_c_rho.png.

Defines rho = ||grad_phi L_eq1|| / ||grad_phi L_eq2|| at network init.

Theoretical prediction (Section 3 of the review):
    Unscaled (1.4): rho = Theta(alpha^{-1})  -> slope -1.
    Scaled (1.5):   rho = Theta(1)          -> slope  0.

Empirically the unscaled slope is often steeper than -1 (e.g. -1.35)
because the loss L_i = (1/2) ||r_i||^2 is quadratic in the residual,
so both the residual and its Jacobian inherit alpha-scaling and the
product compounds. This is noted in the title annotation.
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
OUT_PATH = Path("results") / "exp0_fig_c_rho.png"


def _load():
    data = defaultdict(list)
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["formulation"], float(row["alpha"]))
            data[key].append(float(row["rho"]))
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
        la = np.linspace(log_a.min(), log_a.max(), 50)
        ax.plot(la, intercept + slope * la,
                color=color, lw=0.9, ls=":", alpha=0.7)

    # Reference line rho = 1.
    ax.axhline(0.0, color="black", lw=0.6, ls=":", alpha=0.5)
    ax.annotate(
        r"$\rho = 1$ (balanced)",
        xy=(log_a.min(), 0.0), xytext=(log_a.min() + 0.2, 0.4),
        fontsize=9, color="black",
    )

    ax.set_xlabel(r"$\log_{10}\,\alpha$", fontsize=12)
    ax.set_ylabel(
        r"$\log_{10}\,\rho = \log_{10}"
        r"\|\nabla_\phi L_{\mathrm{eq}_1}\|"
        r"\,/\,\|\nabla_\phi L_{\mathrm{eq}_2}\|$",
        fontsize=12,
    )
    ax.set_title(
        r"Gradient ratio $\rho$ at initialisation"
        "\n"
        r"unscaled slope may exceed $|{-}1|$: quadratic loss compounds "
        r"the Jacobian-block imbalance",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()