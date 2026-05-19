"""
Experiment 8 — Soft BC + Dual Network, α-sensitivity with w_bc sub-sweep.

Sweeps α ∈ {1e-2, 1e-3, 1e-4, 1e-5} under Soft BC with w_bc ∈ {1, 10, 100, 1000}.
Shows that the scaling method's advantage only becomes visible when the
PDE-vs-BC inter-loss imbalance is properly handled (w_bc ≥ 10).

Supports Chinese / English output via labels_zh.LANG toggle.

Output:
    results/exp8_softBC_wbc_sweep.csv         per-run metrics
    results/exp8_softBC_wbc_sweep.png         2×4 panel figure
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
import torch.optim as optim

# ---- labels (Chinese / English) ----
from .labels_zh import LANG, fmt_name, xlabel, ylabel

# ---- device ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"exp8 Soft BC w_bc sweep | device: {device} | lang: {LANG}")

# ---- maths ----
def compute_laplacian(u, x):
    grad_u = torch.autograd.grad(u, x, torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        u_xx = torch.autograd.grad(grad_u[:, i:i+1], x,
                                   torch.ones_like(grad_u[:, i:i+1]),
                                   create_graph=True, retain_graph=True)[0][:, i:i+1]
        lap += u_xx
    return lap

class MMS:
    def __init__(self, alpha):
        self.alpha = alpha
        self.pi = math.pi
    def exact_y(self, x):
        return torch.sin(self.pi * x[:, 0:1]) * torch.sin(self.pi * x[:, 1:2])
    def exact_p(self, x):
        return self.alpha * self.exact_y(x)
    def target_yd(self, x):
        return (1.0 - 2.0 * self.pi**2 * self.alpha) * self.exact_y(x)
    def source_f(self, x):
        return (2.0 * self.pi**2 + 1.0) * self.exact_y(x)
    def prior_ud(self, x):
        return torch.zeros_like(x[:, 0:1])
    def grad_exact_y(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        dy1 = self.pi * torch.cos(self.pi * x1) * torch.sin(self.pi * x2)
        dy2 = self.pi * torch.sin(self.pi * x1) * torch.cos(self.pi * x2)
        return dy1, dy2
    def grad_exact_p(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        a = self.alpha * self.pi
        dp1 = a * torch.cos(self.pi * x1) * torch.sin(self.pi * x2)
        dp2 = a * torch.sin(self.pi * x1) * torch.cos(self.pi * x2)
        return dp1, dp2

# ---- network ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 1),
        )
    def forward(self, x):
        return self.net(x)

# ---- sampling ----
def sample_points(N_int, N_bnd, device):
    xi = torch.rand(N_int, 2, device=device)
    n = N_bnd // 4
    e1 = torch.cat([torch.rand(n, 1), torch.zeros(n, 1)], dim=1)
    e2 = torch.cat([torch.rand(n, 1), torch.ones(n, 1)], dim=1)
    e3 = torch.cat([torch.zeros(n, 1), torch.rand(n, 1)], dim=1)
    e4 = torch.cat([torch.ones(n, 1), torch.rand(n, 1)], dim=1)
    xb = torch.cat([e1, e2, e3, e4], dim=0).to(device)
    return xi, xb

# ---- solver ----
class SoftBCSolver:
    def __init__(self, system_type, alpha, mms, w_bc):
        self.system_type = system_type
        self.alpha = alpha
        self.mms = mms
        self.w_bc = w_bc
        self.net_y = Net().to(device)
        self.net_p = Net().to(device)

    def forward_eval(self, x):
        raw_y, raw_p = self.net_y(x), self.net_p(x)
        if self.system_type == 'scaled':
            return raw_y * (self.alpha ** 0.25), raw_p * (self.alpha ** 0.75)
        return raw_y, raw_p

    def compute_loss(self, xi, xb):
        xi.requires_grad_(True)
        yp, pp = self.forward_eval(xi)
        ly = compute_laplacian(yp, xi)
        lp = compute_laplacian(pp, xi)
        f = self.mms.source_f(xi)
        yd = self.mms.target_yd(xi)
        ud = self.mms.prior_ud(xi)

        if self.system_type == 'unscaled':
            r1 = -ly - (f + ud) + (1.0 / self.alpha) * pp
            r2 = -lp - yp + yd
            lp1 = torch.mean(r1 ** 2)
            lp2 = torch.mean(r2 ** 2)
        else:
            a12 = self.alpha ** 0.5
            a34 = self.alpha ** 0.75
            a14 = self.alpha ** 0.25
            r1 = -a12 * ly + pp - a34 * (f + ud)
            r2 = -a12 * lp - yp + a14 * yd
            lp1 = torch.mean((r1 / a34) ** 2)
            lp2 = torch.mean((r2 / a14) ** 2)

        yb, pb = self.forward_eval(xb)
        if self.system_type == 'scaled':
            lbc = (torch.mean((yb / (self.alpha ** 0.25)) ** 2) +
                   torch.mean((pb / (self.alpha ** 0.75)) ** 2))
        else:
            lbc = torch.mean(yb ** 2) + torch.mean(pb ** 2)

        total = lp1 + lp2 + self.w_bc * lbc
        return total, lp1, lp2, lbc

# ---- training ----
def train(solver, adam_epochs=2000, lbfgs_iters=1000):
    params = list(solver.net_y.parameters()) + list(solver.net_p.parameters())
    opt = optim.Adam(params, lr=1e-3)
    solver.net_y.train()
    solver.net_p.train()
    for _ in range(adam_epochs):
        xi, xb = sample_points(2500, 400, device)
        opt.zero_grad()
        loss, _, _, _ = solver.compute_loss(xi, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)
        opt.step()
    xi_s, xb_s = sample_points(2500, 400, device)
    opt2 = optim.LBFGS(
        params, lr=1.0, max_iter=lbfgs_iters,
        max_eval=int(lbfgs_iters * 1.25),
        history_size=50, tolerance_grad=1e-7, tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    def closure():
        opt2.zero_grad()
        loss, _, _, _ = solver.compute_loss(xi_s, xb_s)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)
        return loss
    try:
        opt2.step(closure)
    except Exception:
        pass

# ---- evaluation ----
def evaluate(solver, mms):
    solver.net_y.eval()
    solver.net_p.eval()
    x1, x2 = torch.meshgrid(
        torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing='ij')
    xt = torch.stack([x1.flatten(), x2.flatten()], dim=-1).to(device)
    xt.requires_grad_(True)
    raw_y, raw_p = solver.forward_eval(xt)
    if solver.system_type == 'scaled':
        yp = raw_y * (solver.alpha ** -0.25)
        pp = raw_p * (solver.alpha ** 0.25)
    else:
        yp, pp = raw_y, raw_p
    ye = mms.exact_y(xt)
    pe = mms.exact_p(xt)
    gy = torch.autograd.grad(yp, xt, torch.ones_like(yp),
                             create_graph=False, retain_graph=True)[0]
    gp = torch.autograd.grad(pp, xt, torch.ones_like(pp),
                             create_graph=False, retain_graph=False)[0]
    dy1, dy2 = mms.grad_exact_y(xt)
    dp1, dp2 = mms.grad_exact_p(xt)
    gye = torch.cat([dy1, dy2], dim=1)
    gpe = torch.cat([dp1, dp2], dim=1)
    ey = yp - ye; ep = pp - pe
    gey = gy - gye; gep = gp - gpe
    return {
        'l2_y': torch.sqrt((ey**2).sum() / (ye**2).sum()).item(),
        'l2_p': torch.sqrt((ep**2).sum() / (pe**2).sum()).item(),
        'linf_y': torch.max(torch.abs(ey)).item(),
        'linf_p': torch.max(torch.abs(ep)).item(),
        'h1_y': torch.sqrt(((ey**2).sum()+(gey**2).sum()) /
                           ((ye**2).sum()+(gye**2).sum())).item(),
        'h1_p': torch.sqrt(((ep**2).sum()+(gep**2).sum()) /
                           ((pe**2).sum()+(gpe**2).sum())).item(),
    }

# ===================== main =====================
alphas = [1e-2, 1e-3, 1e-4, 1e-5]
w_bcs  = [1.0, 10.0, 100.0, 1000.0]

results = {
    w: {s: {m: [] for m in ['l2_y','l2_p','linf_y','linf_p','h1_y','h1_p']}
        for s in ['unscaled','scaled']}
    for w in w_bcs
}

print(f"\n{'w_bc':<6} | {'α':<10} | {'Sys':<10} | {'L2_y':<11} | {'L2_p':<11} | {'Time(s)':<8}")
print("-" * 68)

for w in w_bcs:
    for a in alphas:
        mms = MMS(a)
        for s in ['unscaled', 'scaled']:
            torch.manual_seed(42)
            solver = SoftBCSolver(s, a, mms, w)
            t0 = time.time()
            train(solver)
            el = time.time() - t0
            e = evaluate(solver, mms)
            for k in e:
                results[w][s][k].append(e[k])
            print(f"{w:<6.0f} | {a:<10.0e} | {s:<10} | {e['l2_y']:<11.4e} | {e['l2_p']:<11.4e} | {el:<8.1f}")

# ===================== plot =====================
COLOR_BEFORE = '#0072B2'   # Unscaled
COLOR_AFTER  = '#D55E00'   # Scaled

fig, axes = plt.subplots(2, len(w_bcs), figsize=(5 * len(w_bcs), 9), sharey='row')

if LANG == "zh":
    fig.suptitle(r'Soft BC + 双网络下 $\alpha$-敏感性 — 边界权重 $w_{bc}$ 子扫描', fontsize=15)
else:
    fig.suptitle(r'$\alpha$-Sensitivity under Soft BC + Dual Network, '
                 r'boundary-weight sub-sweep', fontsize=15)

for col, w in enumerate(w_bcs):
    # row 0: L²_y
    ax = axes[0, col]
    ax.loglog(alphas, results[w]['unscaled']['l2_y'], 'o-', color=COLOR_BEFORE,
              lw=2, label=fmt_name('unscaled'))
    ax.loglog(alphas, results[w]['scaled']['l2_y'], 's-', color=COLOR_AFTER,
              lw=2, label=fmt_name('scaled'))
    ax.invert_xaxis()
    ax.set_title(rf'$w_{{\mathrm{{bc}}}} = {int(w)}$')
    ax.set_xlabel(xlabel('alpha'))
    if col == 0:
        ax.set_ylabel(ylabel('l2_y'))
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    # row 1: L²_p
    ax = axes[1, col]
    ax.loglog(alphas, results[w]['unscaled']['l2_p'], 'o-', color=COLOR_BEFORE,
              lw=2, label=fmt_name('unscaled'))
    ax.loglog(alphas, results[w]['scaled']['l2_p'], 's-', color=COLOR_AFTER,
              lw=2, label=fmt_name('scaled'))
    ax.invert_xaxis()
    ax.set_xlabel(xlabel('alpha'))
    if col == 0:
        ax.set_ylabel(ylabel('l2_p'))
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
png = out_dir / "exp8_softBC_wbc_sweep.png"
fig.savefig(png, dpi=150)
plt.close(fig)
print(f"\n→ {png}")

# ---- CSV ----
csv_path = out_dir / "exp8_softBC_wbc_sweep.csv"
with csv_path.open("w", newline="") as fh:
    fieldnames = ['w_bc', 'alpha', 'system', 'l2_y', 'l2_p', 'linf_y', 'linf_p', 'h1_y', 'h1_p']
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for wb in w_bcs:
        for i, a in enumerate(alphas):
            for s in ['unscaled', 'scaled']:
                w.writerow({fn: results[wb][s][fn][i] for fn in fieldnames[3:]}
                           | {'w_bc': wb, 'alpha': a, 'system': s})
print(f"→ {csv_path}")
print("Done.")
