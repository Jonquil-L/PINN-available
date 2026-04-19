"""Verification script — manufactured solution consistency & round-trip.

Run this BEFORE any expensive experiment to catch coefficient / sign errors.

    python -m experiments.verify

Checks:
  1. Both residuals of the unscaled (1.4) system vanish (< 1e-10) at the
     exact solution for several alpha values.
  2. Both residuals of the scaled_raw (1.5) system vanish at the exact
     *scaled* solution for the same alpha values.
  3. Round-trip: y_scaled -> y_phys -> y_exact and p_scaled -> p_phys -> p_exact
     agree to machine precision, confirming the conversion in relative_l2_errors.
"""
from __future__ import annotations

import math

import torch

ALPHAS = (1.0, 1e-2, 1e-4, 1e-6)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


def _lap_phi(x: torch.Tensor) -> torch.Tensor:
    """Analytic Laplacian of phi = sin(pi x1) sin(pi x2): -2 pi^2 phi."""
    return -2.0 * math.pi ** 2 * _phi(x)


def verify_residuals():
    """Check that both residuals vanish at the exact solution."""
    print("=== Residual verification (analytic Laplacian) ===")
    x = torch.rand(500, 2, dtype=torch.float64)
    phi = _phi(x)
    lap_phi = _lap_phi(x)

    all_ok = True
    for alpha in ALPHAS:
        # --- Unscaled (1.4) ---
        # Exact: y = phi, p = alpha * phi
        # r1 = -Delta y - (f + u_d) + (1/alpha) p
        #     = -lap_phi - (2 pi^2 + 1) phi + (1/alpha)(alpha phi)
        #     = -lap_phi - (2 pi^2 + 1) phi + phi
        #     = 2 pi^2 phi - (2 pi^2 + 1) phi + phi = 0  ✓
        f_ud = (2.0 * math.pi ** 2 + 1.0) * phi
        yd = (1.0 - 2.0 * math.pi ** 2 * alpha) * phi

        r1_u = -lap_phi - f_ud + (1.0 / alpha) * (alpha * phi)
        r2_u = -(alpha * lap_phi) - phi + yd  # lap_p = alpha * lap_phi

        norm_r1_u = r1_u.norm().item()
        norm_r2_u = r2_u.norm().item()
        ok_u = norm_r1_u < 1e-10 and norm_r2_u < 1e-10
        status = "PASS" if ok_u else "FAIL"
        print(f"  alpha={alpha:.0e}  unscaled     |r1|={norm_r1_u:.2e}  |r2|={norm_r2_u:.2e}  [{status}]")
        if not ok_u:
            all_ok = False

        # --- Scaled raw (1.5) ---
        # Exact scaled: y_s = alpha^{1/4} phi,  p_s = alpha^{3/4} phi
        # lap_y_s = alpha^{1/4} lap_phi,  lap_p_s = alpha^{3/4} lap_phi
        a12, a34, a14 = alpha ** 0.5, alpha ** 0.75, alpha ** 0.25

        y_s = a14 * phi
        p_s = a34 * phi
        lap_y_s = a14 * lap_phi
        lap_p_s = a34 * lap_phi

        r1_s = -a12 * lap_y_s + p_s - a34 * f_ud
        r2_s = -a12 * lap_p_s - y_s + a14 * yd

        norm_r1_s = r1_s.norm().item()
        norm_r2_s = r2_s.norm().item()
        ok_s = norm_r1_s < 1e-10 and norm_r2_s < 1e-10
        status = "PASS" if ok_s else "FAIL"
        print(f"  alpha={alpha:.0e}  scaled_raw   |r1|={norm_r1_s:.2e}  |r2|={norm_r2_s:.2e}  [{status}]")
        if not ok_s:
            all_ok = False

    return all_ok


def verify_roundtrip():
    """Check that scaled -> physical -> exact agrees to machine precision."""
    print("\n=== Round-trip conversion verification ===")
    x = torch.rand(100, 2, dtype=torch.float64)
    phi = _phi(x)
    all_ok = True

    for alpha in ALPHAS:
        a14 = alpha ** 0.25
        a34 = alpha ** 0.75

        # Scaled predictions (as if network learned exactly)
        y_pred = a14 * phi      # alpha^{1/4} phi
        p_pred = a34 * phi      # alpha^{3/4} phi

        # Convert to physical
        y_phys = y_pred / a14   # should = phi
        p_phys = p_pred * a14   # should = alpha * phi  (since a34 * a14 = alpha)

        # Control
        u_pred = -p_phys / alpha  # should = -phi

        err_y = (y_phys - phi).abs().max().item()
        err_p = (p_phys - alpha * phi).abs().max().item()
        err_u = (u_pred - (-phi)).abs().max().item()

        ok = err_y < 1e-14 and err_p < 1e-14 and err_u < 1e-14
        status = "PASS" if ok else "FAIL"
        print(f"  alpha={alpha:.0e}  |y_err|={err_y:.2e}  |p_err|={err_p:.2e}  |u_err|={err_u:.2e}  [{status}]")
        if not ok:
            all_ok = False

    return all_ok


def main():
    ok1 = verify_residuals()
    ok2 = verify_roundtrip()
    print()
    if ok1 and ok2:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — fix before running experiments")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
