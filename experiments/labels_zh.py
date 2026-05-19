"""
Chinese / English label maps for thesis figures.
Set LANG = "zh" or "en" before importing, or edit the default below.
On Windows, CJK fonts (Microsoft YaHei / SimHei) are auto-configured.
"""
import platform
import matplotlib.pyplot as plt

# ---- toggle ----
LANG = "zh"   # "zh" = 中文,  "en" = English

# ---- CJK font setup (Windows / macOS / Linux) ----
if LANG == "zh":
    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    elif platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ===================== shared label dictionaries =====================

# --- formulation names ---
FORMULATION = {
    "en": {
        "unscaled":   "unscaled (1.4)",
        "scaled_raw": "scaled (1.5)",
        "scaled":     "scaled (1.5)",
    },
    "zh": {
        "unscaled":   "尺度变换前 (式 1.4)",
        "scaled_raw": "尺度变换后 (式 1.5)",
        "scaled":     "尺度变换后 (式 1.5)",
    },
}

# --- alpha-sensitivity y-labels ---
YLABELS = {
    "en": {
        "l2_y":        r"Relative $L^2$ error of $\bar{y}$",
        "l2_p":        r"Relative $L^2$ error of $\bar{p}$",
        "rho_ratio":   r"$\rho = \|\nabla_\phi L_{\mathrm{eq}_1}\| / \|\nabla_\phi L_{\mathrm{eq}_2}\|$",
        "sigma_max":   r"$\log_{10}\,\sigma_{\max}(J)$",
        "block_norm":  r"block Frobenius norm  $\|J_{ij}\|_F$",
        "log_rho":     (r"$\log_{10}\,\rho = \log_{10}"
                        r"\|\nabla_\phi L_{\mathrm{eq}_1}\|"
                        r"\,/\,\|\nabla_\phi L_{\mathrm{eq}_2}\|$"),
        "log_rho_omega": (r"$\log_{10}\,\rho_\omega = "
                          r"\log_{10}(\omega\,\|\nabla L_{\mathrm{eq}_1}\|"
                          r"/\|\nabla L_{\mathrm{eq}_2}\|)$"),
        "abs_log_rho": r"$|\log_{10}\,\rho_\omega|$  (distance from balance)",
    },
    "zh": {
        "l2_y":        r"$\bar{y}$ 的相对 $L^2$ 误差",
        "l2_p":        r"$\bar{p}$ 的相对 $L^2$ 误差",
        "rho_ratio":   r"$\rho = \|\nabla_\phi L_{\mathrm{eq}_1}\| / \|\nabla_\phi L_{\mathrm{eq}_2}\|$",
        "sigma_max":   r"$\log_{10}\,\sigma_{\max}(J)$",
        "block_norm":  r"分块 Frobenius 范数 $\|J_{ij}\|_F$",
        "log_rho":     (r"$\log_{10}\,\rho = \log_{10}"
                        r"\|\nabla_\phi L_{\mathrm{eq}_1}\|"
                        r"\,/\,\|\nabla_\phi L_{\mathrm{eq}_2}\|$"),
        "log_rho_omega": (r"$\log_{10}\,\rho_\omega = "
                          r"\log_{10}(\omega\,\|\nabla L_{\mathrm{eq}_1}\|"
                          r"/\|\nabla L_{\mathrm{eq}_2}\|)$"),
        "abs_log_rho": r"$|\log_{10}\,\rho_\omega|$  (距平衡的距离)",
    },
}

# --- x-label ---
XLABELS = {
    "en": {
        "alpha":       r"regularisation parameter  $\alpha$",
        "log_alpha":   r"$\log_{10}\,\alpha$",
        "log_omega":   r"$\log_{10}\,\omega$",
        "iteration":   "iteration",
    },
    "zh": {
        "alpha":       r"正则化参数 $\alpha$",
        "log_alpha":   r"$\log_{10}\,\alpha$",
        "log_omega":   r"$\log_{10}\,\omega$",
        "iteration":   "迭代步数",
    },
}

# --- common annotations ---
ANNOT = {
    "en": {
        "balanced": r"$\rho = 1$ (balanced)",
        "abs_balanced": r"$|\log\rho|=0$ (balanced)",
    },
    "zh": {
        "balanced": r"$\rho = 1$ (平衡)",
        "abs_balanced": r"$|\log\rho|=0$ (平衡)",
    },
}

# ===================== helper =====================
def _l(dct: dict, key: str) -> str:
    """Look up (key) in dict-of-dicts using current LANG, fallback to en."""
    return dct.get(LANG, dct.get("en", {})).get(key, key)


# convenience accessors — import these
def fmt_name(f: str) -> str:
    return _l(FORMULATION, f)

def ylabel(key: str) -> str:
    return _l(YLABELS, key)

def xlabel(key: str) -> str:
    return _l(XLABELS, key)

def annot(key: str) -> str:
    return _l(ANNOT, key)
