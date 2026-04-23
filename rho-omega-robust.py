"""Principle-level diagnostic: gradient ratio rho vs loss weight omega.

Context in your paper: this diagnostic explains at the PRINCIPLE LEVEL why
the scaled system (1.19) is insensitive to the choice of loss weight omega,
while the unscaled system (1.17) is not. Pairs with your Section 4.4-4.6
phenomenological robustness experiments.

===============================================================================
Setup
===============================================================================
Fix alpha = 1e-4 (stiff regime where unscaled is already pathological).
Sweep omega in {1e-2, 1e-1, 1, 1e1, 1e2}.
Total loss actually used to train: L(phi; omega) = omega * L_1 + L_2
  with L_i = mean(r_i^2) in the usual PINN way.

Diagnostic reported:
    rho(phi; omega) = ||grad_phi (omega * L_1)|| / ||grad_phi L_2||

A "balanced" rho lies in [0.1, 10]. The smaller the range of omega for which
rho stays balanced, the more hyperparameter-sensitive the formulation is.

===============================================================================
Measurement point: post-warmup, not initialisation
===============================================================================
The naive "at initialisation" measurement is MISLEADING for this problem:
an untrained network with Xavier init produces D(x)*N(x) of order O(1),
regardless of the formulation. But the scaled system's theoretical block
balance (Gong-Tan-Zhou 2022 Theorem 3.4) assumes y_s = alpha^{1/4} phi and
p_s = alpha^{3/4} phi, i.e. the OPTIMAL scaling. An untrained network
does not yet know this, so r1_raw is dominated by its largest term
(p_s = O(1)), and after the 1/alpha^{3/4} normalisation you end up with
an artificial O(alpha^{-3/4}) residual that has nothing to do with
conditioning of the converged system.

Fix: run 500 Adam steps as warmup, then measure rho. After warmup, the
networks have converged to the natural scale of their formulation, and
the reported rho faithfully reflects the CONDITIONING of the loss landscape
the optimiser will experience during most of training. This is the
quantity that matters for robustness claims.

===============================================================================
Manufactured problem (matches Report.tex Section 4.1)
===============================================================================
Omega = (0,1)^2, phi(x) = sin(pi x_1) sin(pi x_2)
  y_bar_star = phi;  p_bar_star = alpha * phi
  f + u_d    = (2 pi^2 + 1) phi
  y_d        = (1 - 2 pi^2 alpha) phi

Network:       4 hidden layers x 50 units, tanh. Hard BC via
               D(x) = x_1 (1 - x_1) x_2 (1 - x_2) ansatz.
Seeds:         10
Warmup steps:  500 Adam, lr = 1e-3, total loss = L1 + L2 (equal weights)
Precision:     float64 on CPU

===============================================================================
Outputs
===============================================================================
  results/rho_omega.csv
  results/rho_omega.png

Runtime: ~3-5 minutes on a laptop CPU.
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
import torch.nn as nn


ALPHA = 1e-4
OMEGAS = (1e-2, 1e-1, 1.0, 1e1, 1e2)
SEEDS = tuple(range(10))

NET_WIDTH = 50
NET_DEPTH = 4
N_COLLOCATION = 2500

WARMUP_STEPS = 500
WARMUP_LR = 1e-3

DTYPE = torch.float64
DEVICE = torch.device("cuda")

OUT_DIR = Path("results")


def manufactured_data(x: torch.Tensor, alpha: float):
    pi = math.pi
    phi = torch.sin(pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
    fud = (2.0 * pi * pi + 1.0) * phi
    yd = (1.0 - 2.0 * pi * pi * alpha) * phi
    return fud, yd


def sample_interior(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=DTYPE, device=DEVICE)
    x.requires_grad_(True)
    return x


class MLP(nn.Module):
    def __init__(self, width: int = NET_WIDTH, depth: int = NET_DEPTH):
        super().__init__()
        layers = [nn.Linear(2, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers[:-1]:
            x = torch.tanh(lin(x))
        return self.layers[-1](x)


def boundary_factor(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])


def build_net(seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP().to(device=DEVICE, dtype=DTYPE)


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    gradu = torch.autograd.grad(
        u.sum(), x, create_graph=True, retain_graph=True
    )[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        d2u = torch.autograd.grad(
            gradu[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0][:, i : i + 1]
        lap = lap + d2u
    return lap


def residuals_unscaled(net_y: MLP, net_p: MLP, x: torch.Tensor, alpha: float):
    D = boundary_factor(x)
    y = D * net_y(x)
    p = D * net_p(x)
    lap_y = laplacian(y, x)
    lap_p = laplacian(p, x)
    fud, yd = manufactured_data(x, alpha)
    r1 = -lap_y + (1.0 / alpha) * p - fud
    r2 = -lap_p - y + yd
    return r1, r2


def residuals_scaled(net_y: MLP, net_p: MLP, x: torch.Tensor, alpha: float):
    a12 = alpha ** 0.5
    a34 = alpha ** 0.75
    a14 = alpha ** 0.25
    D = boundary_factor(x)
    y_s = D * net_y(x)
    p_s = D * net_p(x)
    lap_ys = laplacian(y_s, x)
    lap_ps = laplacian(p_s, x)
    fud, yd = manufactured_data(x, alpha)
    r1_raw = -a12 * lap_ys + p_s - a34 * fud
    r2_raw = -a12 * lap_ps - y_s + a14 * yd
    r1 = r1_raw / a34
    r2 = r2_raw / a14
    return r1, r2


def residuals(formulation: str, net_y, net_p, x, alpha):
    if formulation == "unscaled":
        return residuals_unscaled(net_y, net_p, x, alpha)
    if formulation == "scaled":
        return residuals_scaled(net_y, net_p, x, alpha)
    raise ValueError(formulation)


def warmup(net_y, net_p, formulation, alpha, seed):
    params = list(net_y.parameters()) + list(net_p.parameters())
    opt = torch.optim.Adam(params, lr=WARMUP_LR)
    x = sample_interior(N_COLLOCATION, seed=seed + 30_000)
    for _ in range(WARMUP_STEPS):
        opt.zero_grad()
        r1, r2 = residuals(formulation, net_y, net_p, x, alpha)
        loss = (r1 ** 2).mean() + (r2 ** 2).mean()
        loss.backward()
        opt.step()


def flat_grad_norm(loss, params, retain_graph: bool) -> float:
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.detach().pow(2).sum().item())
    return math.sqrt(sq)


def compute_rho(formulation: str, alpha: float, omega: float, seed: int) -> dict:
    net_y = build_net(seed=seed)
    net_p = build_net(seed=seed + 10_000)
    warmup(net_y, net_p, formulation, alpha, seed)

    x = sample_interior(N_COLLOCATION, seed=seed + 40_000)
    r1, r2 = residuals(formulation, net_y, net_p, x, alpha)
    L1 = (r1 ** 2).mean()
    L2 = (r2 ** 2).mean()

    params = list(net_y.parameters()) + list(net_p.parameters())
    g1 = flat_grad_norm(omega * L1, params, retain_graph=True)
    g2 = flat_grad_norm(L2, params, retain_graph=False)
    rho = g1 / max(g2, 1e-300)
    return {
        "rho": rho,
        "grad_L1_weighted": g1,
        "grad_L2": g2,
        "L1": float(L1.detach()),
        "L2": float(L2.detach()),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    rows = []

    for formulation in ("unscaled", "scaled"):
        for omega in OMEGAS:
            rho_vals = []
            for seed in SEEDS:
                r = compute_rho(formulation, ALPHA, omega, seed)
                rho_vals.append(r["rho"])
            rho_arr = np.asarray(rho_vals)
            log_rho = np.log10(rho_arr)
            row = {
                "formulation": formulation,
                "alpha": ALPHA,
                "omega": omega,
                "rho_mean": float(rho_arr.mean()),
                "rho_median": float(np.median(rho_arr)),
                "log10_rho_mean": float(log_rho.mean()),
                "log10_rho_std": float(log_rho.std()),
                "log10_rho_p10": float(np.percentile(log_rho, 10)),
                "log10_rho_p90": float(np.percentile(log_rho, 90)),
            }
            rows.append(row)
            print(
                f"{formulation:<9s} omega={omega:.0e}  "
                f"rho = 10^{row['log10_rho_mean']:+.2f} "
                f"(sd 10^{row['log10_rho_std']:.2f})"
            )

    print(f"\nTotal sweep time: {time.time() - t0:.1f} s")

    csv_path = OUT_DIR / "rho_omega.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = {"unscaled": ("tomato", "o"), "scaled": ("steelblue", "s")}
    for formulation in ("unscaled", "scaled"):
        data = [r for r in rows if r["formulation"] == formulation]
        data.sort(key=lambda r: r["omega"])
        log_om = np.log10(np.array([r["omega"] for r in data]))
        mu = np.array([r["log10_rho_mean"] for r in data])
        sd = np.array([r["log10_rho_std"] for r in data])
        slope, intercept = np.polyfit(log_om, mu, 1)
        color, marker = palette[formulation]
        ax.errorbar(
            log_om, mu, yerr=sd, fmt=f"{marker}-", color=color, capsize=4, lw=1.8,
            label=f"{formulation}  (slope {slope:+.2f}, intercept {intercept:+.2f})",
        )

    ax.axhspan(-1, 1, color="lightgreen", alpha=0.25, zorder=0,
               label=r"balanced band: $|\log_{10}\rho| \leq 1$")
    ax.axhline(0, color="green", lw=0.8, ls=":")

    ax.set_xlabel(r"$\log_{10} \omega$")
    ax.set_ylabel(r"$\log_{10} \rho$ (gradient norm ratio)")
    ax.set_title(
        r"Gradient ratio $\rho$ vs loss weight $\omega$"
        + f"\n(alpha = 1e-4, post-warmup, {len(SEEDS)} seeds, 4x50 tanh net)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    png_path = OUT_DIR / "rho_omega.png"
    plt.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    def in_band(r):
        return -1.0 <= r["log10_rho_mean"] <= 1.0

    for formulation in ("unscaled", "scaled"):
        data = [r for r in rows if r["formulation"] == formulation]
        in_count = sum(1 for r in data if in_band(r))
        data.sort(key=lambda r: r["omega"])
        log10_range = (
            min(r["log10_rho_mean"] for r in data),
            max(r["log10_rho_mean"] for r in data),
        )
        print(
            f"  {formulation:<9s}: log10(rho) spans [{log10_range[0]:+.2f}, "
            f"{log10_range[1]:+.2f}];  "
            f"{in_count}/{len(OMEGAS)} omega values in balanced band"
        )


if __name__ == "__main__":
    main()


