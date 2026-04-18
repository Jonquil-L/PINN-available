# PINN-available: Scaled vs Unscaled PINN for the Poisson OCP

Numerical experiments that compare the **unscaled KKT system (1.4)** with the
**symmetric-√α scaled KKT system (1.5)** for the distributed-control
Poisson optimal control problem on Ω = (0,1)².

Each experiment isolates one mechanism that (1.5) is claimed to improve:

| # | File | Mechanism |
|---|------|-----------|
| 1 | `experiments/exp1_conditioning.py`      | Jacobian conditioning: κ(JᵀJ) vs α — **quick variant** (small net 2×20, dense SVD) |
| 1b | `experiments/exp1_conditioning_v2.py`  | Jacobian conditioning — **spec-faithful variant** (full 4×50 net, randomized SVD, 5 seeds) |
| 2 | `experiments/exp2_gradient_balance.py`  | Gradient ratio ρ = ‖∇L₁‖/‖∇L₂‖ during training |
| 3 | `experiments/exp3_stable_lr.py`         | Maximum stable Adam learning rate |
| 4 | `experiments/exp4_float_stability.py`   | Float32 vs float64 term contributions |
| 5 | `experiments/exp5_accuracy.py`          | End-to-end relative L² error on y, p, u |
| 6 | `experiments/exp6_adaptive_ablation.py` | (1.4) + Wang et al. adaptive weights vs (1.5) |

## Problem

Distributed-control Poisson OCP:
```
min   ½ ‖y - y_d‖² + (α/2) ‖u - u_d‖²
s.t.  -Δy = f + u   in Ω
       y = 0        on ∂Ω
```
With KKT conditions and `u = u_d - p/α`, the two formulations compared are:

**Unscaled (1.4):**
```
r₁ = -Δy - (f + u_d) + (1/α) p
r₂ = -Δp -  y        +       y_d
```

**Scaled (1.5) (symmetric √α):**
```
r₁_s / α^(3/4) = (-α^(1/2) Δy + p - α^(3/4)(f + u_d)) / α^(3/4)
r₂_s / α^(1/4) = (-α^(1/2) Δp - y + α^(1/4) y_d)     / α^(1/4)
```

Manufactured solution (consistent with the KKT system so both residuals
vanish exactly at the truth):
```
y*(x) = sin(π x₁) sin(π x₂)
p*(x) = α sin(π x₁) sin(π x₂)
u*(x) = -sin(π x₁) sin(π x₂)
```

## Install

### Windows 11 + CUDA 12.9 (recommended)

Requires **Python 3.10** and an NVIDIA GPU with **CUDA 12.9** drivers.

**Option A — one-click script:**
```cmd
install_win.bat
```

**Option B — manual:**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install "numpy>=1.24,<2.0" "scipy>=1.10" "matplotlib>=3.7"
```

Verify CUDA is visible:
```cmd
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### macOS / Linux

```bash
pip install -r requirements.txt
```
On macOS the PyPI wheel is CPU+MPS; on Linux you may want the CUDA
index: `pip install torch --index-url https://download.pytorch.org/whl/cu129`.

## Running

Device is auto-selected in this order: **CUDA → MPS → CPU**. Override with
the env var `PINN_DEVICE=cpu|mps|cuda` (see `common.pick_device`).

Run each experiment from the repo root:

```bash
python -m experiments.exp1_conditioning     # quick variant (small net, dense SVD)
python -m experiments.exp1_conditioning_v2  # spec variant (full net, Gram eigvalsh)
python -m experiments.exp2_gradient_balance
python -m experiments.exp3_stable_lr
python -m experiments.exp4_float_stability
python -m experiments.exp5_accuracy
python -m experiments.exp6_adaptive_ablation
```

Each experiment writes its outputs (CSV + PNG) into `results/`.

### Platform notes

| Platform | Device | float64 support | Notes |
|----------|--------|----------------|-------|
| Windows 11 + CUDA 12.9 | `cuda` | full | Primary target of this branch |
| macOS Apple Silicon | `mps` | partial → Exp 4 falls back to CPU | Auto-detected after CUDA |
| Linux + CUDA | `cuda` | full | Works identically to Windows |
| CPU-only | `cpu` | full | Fallback; slower but fully functional |

## Common setup (shared across all experiments)

- Domain Ω = (0,1)², Latin-hypercube interior samples (`N_r = 2500`).
- Two separate MLPs (one for y, one for p), 4 hidden layers × 50 units,
  `tanh` activation, Xavier/Glorot initialization.
- **Hard boundary condition ansatz** `ỹ(x) = x₁(1-x₁) x₂(1-x₂) · Net(x)`
  (same for p̃). No boundary-loss term is needed.
- Adam optimizer. 20k iterations where applicable (Exp 1 uses 0/1000 for
  snapshots; Exp 2 uses 10k; Exp 3 does a full 20k per LR).
- 5 seeds by default (3 for Exp 1 to keep dense SVD fast). The α sweep is
  `{1, 1e-2, 1e-4, 1e-6, 1e-8}`.

## Expected results

- Exp 1: slope of `log κ(JᵀJ)` vs `log α` ≈ **−2** for (1.4), **−1** for (1.5).
- Exp 2: ρ stays close to 1 for (1.5) across all α; for (1.4), ρ is off
  by roughly α⁻².
- Exp 3: `η_max` scales as α² for (1.4) (slope +2), is α-independent
  (slope 0) for (1.5).
- Exp 4: at α = 1e-6 in float32 for (1.4), the ‖p/α‖² term dominates and
  other terms fall below machine-ε. All terms remain O(1) for (1.5).
- Exp 5: (1.4) L² error diverges for α ≲ 1e-3; (1.5) stays near 1e-3
  across the full range.
- Exp 6: (1.4)+adaptive improves over (1.4) but still degrades at very
  small α; (1.5) matches or beats it with no adaptive tuning.

## Repo layout

```
PINN-available/
├── experiments/
│   ├── common.py                 # shared modules
│   ├── exp1_conditioning.py         # quick variant
│   ├── exp1_conditioning_v2.py      # full-net + randomized SVD
│   ├── exp2_gradient_balance.py
│   ├── exp3_stable_lr.py
│   ├── exp4_float_stability.py
│   ├── exp5_accuracy.py
│   └── exp6_adaptive_ablation.py
├── requirements.txt
└── README.md
```
