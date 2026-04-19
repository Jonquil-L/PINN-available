# Experiment 3 Figure Notes (exp3_eta_max)

## 1) Output files
- Figure: `results/exp3_eta_max.png`
- Data: `results/exp3_eta_max.csv`
- Script reference: `experiments/exp3_stable_lr.py`

## 2) What this figure shows
The figure is a log-log plot of:
- x-axis: alpha
- y-axis: eta_max (largest stable Adam learning rate)

Two formulations are compared:
- `unscaled`
- `scaled_raw`

For each formulation, the curve is built from CSV values and a linear fit is applied in log10-space:

log10(eta_max) = m * log10(alpha) + b

where `m` is shown in the legend as the slope.

## 3) Data used (from CSV)
- unscaled: (1, 1e-1), (1e-2, 1e-1), (1e-4, 1e-3), (1e-6, 3e-3), (1e-8, 1e-3)
- scaled_raw: (1, 3e-2), (1e-2, 1e-1), (1e-4, 3e-5), (1e-6, 3e-5), (1e-8, 3e-5)

## 4) Fitted slopes from the current CSV
- unscaled slope: 0.2761
- scaled_raw slope: 0.4761

These values are exactly what is plotted from the current CSV file.

## 5) Why this may differ from the theoretical trend
The README expectation for Experiment 3 is:
- unscaled: slope about +2
- scaled formulation: slope about 0

Current observed slopes differ, likely due to one or more of:
- single-seed pass (`seed=0`) in the run
- coarse LR sweep grid
- strict binary stability criterion (end-loss + moving-window monotonicity)
- non-monotone eta_max outcomes at small alpha

So this figure should be interpreted as the result of this specific run configuration, not yet as a final statistical estimate.

## 6) How the figure was generated in this workspace
The PNG was regenerated directly from CSV using the same plotting logic as `exp3_stable_lr.py`:
- log-log curves
- x-axis inverted
- legend with per-series fitted slope

No additional training run was required.

## 7) Recommended next step (if you want publication-grade trend)
- run multiple seeds per (formulation, alpha, lr)
- define eta_max from a success-rate threshold (for example >= 80%)
- optionally densify LR sweep near the transition boundary
- then refit slope on aggregated eta_max
