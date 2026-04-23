"""Plot 1 — Block Frobenius norms of J vs alpha.

Reads results/exp0_data.csv (produced by exp0_collect.py) and writes
results/exp0_fig_a_block_norms.png.

Theoretical prediction:
    Unscaled (1.4): J12 ~ alpha^{-1} (slope +1 on inverted x-axis),
                    J11, J21, J22 ~ O(1)  (slope 0).
    Scaled (1.5):   J11, J22 ~ alpha^{1/2} (slope -1/2 on inverted axis),
                    J12, J21 ~ O(1).

Every curve is labeled. Median over seeds; shaded band = 10-90% quantiles.
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
OUT_PATH = Path("results") / "exp0_fig_a_block_norms.png"


def _load():
    """Return {(formulation, alpha, block): [values]}."""
    data = defaultdict(list)
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            formulation = row["formulation"]
            alpha = float(row["alpha"])
            for bk in ("J11", "J12", "J21", "J22"):
                data[(formulation, alpha, bk)].append(float(row[bk]))
    return data


def _summary(vals):
    v = np.array(vals)
    return np.median(v), np.quantile(v, 0.1), np.quantile(v, 0.9)


def main():
    data = _load()
    alphas = sorted({k[1] for k in data.keys()}, reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color per block index, linestyle per formulation.
    block_info = [
        ("J11", r"$\|J_{11}\|_F$  (eq 1 / $y$-net)", "C0"),
        ("J12", r"$\|J_{12}\|_F$  (eq 1 / $p$-net)", "C1"),
        ("J21", r"$\|J_{21}\|_F$  (eq 2 / $y$-net)", "C2"),
        ("J22", r"$\|J_{22}\|_F$  (eq 2 / $p$-net)", "C3"),
    ]
    fstyle = {
        "unscaled":   {"ls": "-",  "marker": "o", "suffix": "(1.4) unscaled"},
        "scaled_raw": {"ls": "--", "marker": "s", "suffix": "(1.5) scaled"},
    }

    a_arr = np.array(alphas)

    for bk, block_label, color in block_info:
        for formulation in ("unscaled", "scaled_raw"):
            s = fstyle[formulation]
            med, lo, hi = [], [], []
            for alpha in alphas:
                vals = data[(formulation, alpha, bk)]
                m, l, h = _summary(vals)
                med.append(m); lo.append(l); hi.append(h)
            med = np.array(med); lo = np.array(lo); hi = np.array(hi)

            ax.fill_between(a_arr, lo, hi, color=color, alpha=0.10)
            ax.plot(
                a_arr, med,
                linestyle=s["ls"], marker=s["marker"],
                color=color, lw=1.8, ms=7,
                label=f"{block_label}  {s['suffix']}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"regularisation parameter  $\alpha$", fontsize=12)
    ax.set_ylabel(r"block Frobenius norm  $\|J_{ij}\|_F$", fontsize=12)
    ax.set_title(
        r"Block Frobenius norms of the PINN Jacobian at initialisation"
        "\n"
        r"solid = unscaled (1.4), dashed = scaled (1.5); median over 10 seeds, band = 10–90\%",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()