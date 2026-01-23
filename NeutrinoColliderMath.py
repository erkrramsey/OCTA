#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeutrinoColliderMath — single-file utility for ε-extraction under quadratic (Majorana/Hyperknot) vs linear scaling.

===============================================================================
CORE MAPPING (WHAT THIS SCRIPT DOES)
===============================================================================

If a ΔL=2 collider signal yield depends quadratically on an effective coupling ε:

    Quadratic (Majorana/Hyperknot-like):    S(ε) = ε² S0

and your signal template (or reference yield) is normalized to ε=1 (i.e., the
unsuppressed rate for that topology/benchmark), then the usual signal strength μ
relative to that template is:

    μ = S / S0 = ε²   =>   ε = √μ

Therefore a 95% CL bound can be expressed as:

    ε95 = √(S95 / S0)   (equivalently ε95 = √μ95)

For a linear benchmark scaling axis:

    Linear benchmark:                     S(ε) = ε S0

the corresponding 95% CL bound is:

    ε95 = S95 / S0

===============================================================================
S95 PROJECTION METHODS (FOR QUICK STUDIES)
===============================================================================

1) Gaussian (background-dominated):
    S95 ≈ 1.64 √(B + δB²)

2) Poisson exact inversion (background known, δB=0), with an assumed observation
   convention for projections:
      - n_obs_mode="floor":  n_obs = floor(B)
      - n_obs_mode="round":  n_obs = round(B)
      - n_obs_mode="asimov": n_obs = int(B + 0.5)  (simple median-ish proxy)

   Solve for s such that:
      P(N ≤ n_obs | λ=B+s) = 0.05   (one-sided 95% CL)

3) Auto:
   Use Poisson when (B < threshold and δB==0), else Gaussian.

WARNING:
  For publication-grade limits, import S95 or μ95 from your collaboration’s CLs /
  profile-likelihood machinery. This script is designed for defensible, low-friction
  projections and cross-checks.

===============================================================================
COPY-PASTE BLURB (WHITEPAPER / NOTE)
===============================================================================

"When the signal template is normalized to the ε=1 (unsuppressed) rate for a quadratic
ΔL=2 model, the fitted signal strength satisfies μ = ε², hence ε95 = √μ95 = √(S95/S0),
where S95 is the background-only 95% CL upper limit on signal events and S0 is the
expected ε=1 yield. For a linear benchmark scaling S(ε)=εS0, one obtains ε95=S95/S0.
Plotting both versus luminosity provides a model-independent visualization of the
small-ε regime in which quadratic suppression materially alters reach."

===============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Poisson utilities (low-B exact inversion)
# =============================================================================

def _poisson_cdf(k: int, lam: float) -> float:
    """
    Poisson CDF P(N <= k | lambda=lam) computed by summing terms.

    Notes:
      • For our intended regime (lam ~ O(1–50), k ~ O(1–50)), this is stable.
      • Guard included for very large lam to avoid exp(-lam) underflow pathologies.
    """
    if k < 0:
        return 0.0
    if lam < 0:
        raise ValueError("lam must be >= 0")

    # Underflow guard (exp(-lam) underflows near lam ~ 700 in double precision)
    # If lam is enormous, CDF is essentially a step around k ~ lam.
    if lam > 700:
        return 0.0 if k < lam else 1.0

    term = math.exp(-lam)  # i=0
    s = term
    for i in range(1, k + 1):
        term *= lam / i
        s += term
    return min(max(s, 0.0), 1.0)


def _select_n_obs(B: float, n_obs_mode: str) -> int:
    m = n_obs_mode.lower().strip()
    if m == "floor":
        return int(math.floor(B))
    if m == "round":
        return int(round(B))
    if m == "asimov":
        # Simple median-ish proxy without introducing non-integer "Asimov counts"
        return int(B + 0.5)
    raise ValueError("n_obs_mode must be one of: 'floor', 'round', 'asimov'")


def s95_poisson_known_bkg(B: float, cl: float = 0.95, n_obs_mode: str = "floor") -> float:
    """
    Exact one-sided 95% CL upper limit on s with known background B (δB=0),
    using an assumed observation n_obs determined by n_obs_mode.

    Solve:
      P(N <= n_obs | λ = B + s) = 1 - cl
    """
    if B < 0:
        raise ValueError("B must be >= 0")
    if not (0.0 < cl < 1.0):
        raise ValueError("cl must be in (0,1)")

    n_obs = _select_n_obs(B, n_obs_mode)
    target = 1.0 - cl

    def f(s: float) -> float:
        return _poisson_cdf(n_obs, B + s) - target  # monotone decreasing in s

    lo = 0.0
    hi = max(5.0, 5.0 * math.sqrt(B + 1.0) + 5.0)

    # Bracket root
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            return hi

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    return max(0.0, 0.5 * (lo + hi))


# =============================================================================
# Gaussian S95 (moderate/large B)
# =============================================================================

def s95_gauss(B: float, dB: float = 0.0) -> float:
    """
    Fast rule-of-thumb (one-sided 95%):
      S95 ≈ 1.64 * sqrt(B + dB^2)
    """
    if B < 0 or dB < 0:
        raise ValueError("B and dB must be >= 0")
    return 1.64 * math.sqrt(B + dB * dB)


def s95(B: float,
        dB: float = 0.0,
        method: str = "auto",
        threshold: float = 10.0,
        n_obs_mode: str = "floor") -> Tuple[float, str]:
    """
    Unified S95 selection.

    method:
      - "gauss"   -> Gaussian approximation
      - "poisson" -> exact Poisson inversion (requires dB=0)
      - "auto"    -> Poisson if (B < threshold and dB==0), else Gaussian

    n_obs_mode applies only to Poisson path.
    """
    method = method.lower().strip()
    if method == "gauss":
        return s95_gauss(B, dB), "gauss"

    if method == "poisson":
        if dB != 0.0:
            raise ValueError("Poisson method assumes known background (dB=0). Use gauss/auto otherwise.")
        return s95_poisson_known_bkg(B, cl=0.95, n_obs_mode=n_obs_mode), f"poisson({n_obs_mode})"

    if method == "auto":
        if (B < threshold) and (dB == 0.0):
            return s95_poisson_known_bkg(B, cl=0.95, n_obs_mode=n_obs_mode), f"poisson({n_obs_mode})"
        return s95_gauss(B, dB), "gauss"

    raise ValueError("method must be one of: 'auto', 'gauss', 'poisson'")


# =============================================================================
# ε extraction
# =============================================================================

def eps95_quadratic(S95: float, S0: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """
    Vectorized ε95 for quadratic scaling:
      ε95 = sqrt(S95/S0)
    """
    if np.any(S0 <= 0):
        raise ValueError("All S0 values must be > 0")
    ratio = np.maximum(S95 / S0, 0.0)
    out = np.sqrt(ratio)
    return np.maximum(out, floor)


def eps95_linear(S95: float, S0: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """
    Vectorized ε95 for linear scaling:
      ε95 = S95/S0
    """
    if np.any(S0 <= 0):
        raise ValueError("All S0 values must be > 0")
    out = np.maximum(S95 / S0, 0.0)
    return np.maximum(out, floor)


# =============================================================================
# Plot + table helpers
# =============================================================================

def plot_sensitivity(
    lumi_ab: Sequence[float],
    B: float,
    S0_per_ab: float,
    *,
    dB: float = 0.0,
    method: str = "auto",
    threshold: float = 10.0,
    n_obs_mode: str = "floor",
    outfile: str = "epsilon_sensitivity.png",
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot ε95 vs luminosity for quadratic vs linear scaling.

    Inputs:
      lumi_ab: list of luminosities in ab^-1
      B: expected background events in SR
      S0_per_ab: expected ε=1 signal yield per ab^-1 (after cuts)
      dB: background uncertainty (Gaussian only)
      method: auto|gauss|poisson
      threshold: auto Poisson crossover
      n_obs_mode: floor|round|asimov (Poisson path)
    """
    L = np.array(list(lumi_ab), dtype=float)
    if L.size == 0 or np.any(L <= 0):
        raise ValueError("lumi_ab must be non-empty and all values > 0")
    if B < 0 or S0_per_ab <= 0:
        raise ValueError("B must be >= 0 and S0_per_ab must be > 0")

    S95, used = s95(B, dB=dB, method=method, threshold=threshold, n_obs_mode=n_obs_mode)

    S0 = S0_per_ab * L  # ε=1 yields
    eq = eps95_quadratic(S95, S0)
    el = eps95_linear(S95, S0)

    plt.figure(figsize=(7.2, 5.0))
    plt.loglog(L, eq, "o-", label="Quadratic: S=ε² S0")
    plt.loglog(L, el, "s--", label="Linear: S=ε S0")
    plt.xlabel(r"Integrated luminosity [ab$^{-1}$]")
    plt.ylabel(r"95% CL bound on $\varepsilon$")
    if title is None:
        title = f"ε scaling ({used}; B={B:g}, S0/ab={S0_per_ab:g})"
    plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

    return {
        "outfile": outfile,
        "S95": float(S95),
        "method_used": used,
        "lumi_ab": L.tolist(),
        "eps95_quadratic": eq.tolist(),
        "eps95_linear": el.tolist(),
    }


def benchmark_table(
    benchmarks: Dict[str, Dict[str, float]],
    lumi_ab: Sequence[float],
    *,
    method: str = "auto",
    threshold: float = 10.0,
    n_obs_mode: str = "floor",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, cfg in benchmarks.items():
        B = float(cfg.get("bkg", 0.0))
        S0_per_ab = float(cfg.get("S0_per_ab", 0.0))
        dB = float(cfg.get("dB", 0.0))
        if S0_per_ab <= 0:
            raise ValueError(f"Benchmark '{name}' needs S0_per_ab > 0")

        S95, used = s95(B, dB=dB, method=method, threshold=threshold, n_obs_mode=n_obs_mode)

        for L in lumi_ab:
            Lf = float(L)
            if Lf <= 0:
                raise ValueError("All luminosities must be > 0")
            S0 = np.array([S0_per_ab * Lf], dtype=float)
            eps_q = float(eps95_quadratic(S95, S0)[0])
            eps_l = float(eps95_linear(S95, S0)[0])
            rows.append({
                "benchmark": name,
                "L_ab": Lf,
                "B": B,
                "dB": dB,
                "S0_per_ab": S0_per_ab,
                "S95": float(S95),
                "method_used": used,
                "eps95_quadratic": eps_q,
                "eps95_linear": eps_l,
            })
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    w = max(len(r["benchmark"]) for r in rows)
    print(f"{'benchmark'.ljust(w)}  {'L':>6}  {'B':>5}  {'S0/ab':>7}  {'S95':>7}  {'eps_quad':>9}  {'eps_lin':>9}  {'method':>14}")
    print("-" * (w + 64))
    for r in rows:
        print(
            f"{r['benchmark'].ljust(w)}  {r['L_ab']:6.1f}  {r['B']:5.1f}  {r['S0_per_ab']:7.1f}  "
            f"{r['S95']:7.2f}  {r['eps95_quadratic']:9.4g}  {r['eps95_linear']:9.4g}  {r['method_used']:>14}"
        )


def latex_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "% (no rows)\n"
    # Group by benchmark
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["benchmark"], []).append(r)
    for k in by:
        by[k] = sorted(by[k], key=lambda x: float(x["L_ab"]))

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Benchmark & $L$ [ab$^{-1}$] & $B$ & $S_{95}$ & $S_0$/ab & $\varepsilon_{95}^{(\mathrm{quad})}$ & $\varepsilon_{95}^{(\mathrm{lin})}$ \\")
    lines.append(r"\hline")
    for bench, rr in by.items():
        for r in rr:
            lines.append(
                f"{bench} & {r['L_ab']:.1f} & {r['B']:.1f} & {r['S95']:.2f} & {r['S0_per_ab']:.1f} & {r['eps95_quadratic']:.4g} & {r['eps95_linear']:.4g} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Projected 95\% CL bounds on $\varepsilon$ under quadratic vs linear scaling.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# =============================================================================
# Default benchmarks (edit freely)
# =============================================================================

DEFAULT_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "10 TeV optimistic": {"bkg": 5.0, "S0_per_ab": 50.0, "dB": 0.0},
    "10 TeV realistic":  {"bkg": 20.0, "S0_per_ab": 100.0, "dB": 0.0},
    "3 TeV early":       {"bkg": 10.0, "S0_per_ab": 20.0, "dB": 0.0},
    "10 TeV slide demo": {"bkg": 8.0, "S0_per_ab": 80.0, "dB": 0.0},
}


# =============================================================================
# Demo run (safe to delete)
# =============================================================================

if __name__ == "__main__":
    # 1) Slide-worthy plot
    out = plot_sensitivity(
        lumi_ab=[1, 3, 10, 30, 100],
        B=8.0,
        S0_per_ab=80.0,
        method="auto",
        threshold=10.0,
        n_obs_mode="floor",  # change to 'round' or 'asimov' if desired
        outfile="10TeV_B8_S0_80.png",
        title="10 TeV demo: B=8, S0/ab=80 (auto S95)",
    )
    print("Wrote:", out["outfile"])
    print("S95:", f"{out['S95']:.4f}", "method:", out["method_used"])

    # 2) Table
    rows = benchmark_table(
        DEFAULT_BENCHMARKS,
        lumi_ab=[1, 3, 10, 30, 100],
        method="auto",
        threshold=10.0,
        n_obs_mode="floor",
    )
    print_table(rows)

    # 3) LaTeX snippet (optional)
    # print(latex_table(rows))import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Stats core
# =============================================================================

def _poisson_cdf(k: int, lam: float) -> float:
    """
    Poisson CDF P(N <= k | lambda=lam) computed by summing terms.

    Safe for our target regime (k ~ floor(B), B usually <= O(50)).
    If you later extend to B >> 100, consider scipy.stats.poisson.cdf.
    """
    if k < 0:
        return 0.0
    if lam < 0:
        raise ValueError("lambda must be >= 0")

    # term0 = e^{-lam}
    term = math.exp(-lam)
    s = term
    for i in range(1, k + 1):
        term *= lam / i
        s += term
    return min(max(s, 0.0), 1.0)


def s95_poisson_background_known(b: float, cl: float = 0.95) -> float:
    """
    Exact one-sided upper limit on s for a counting experiment with known background b,
    using n_obs=floor(b) as an Asimov-ish / typical observation for projections.

    Solve for s such that:
        P(N <= n_obs | lambda = b + s) = 1 - cl
    """
    if b < 0:
        raise ValueError("b must be >= 0")
    if not (0.0 < cl < 1.0):
        raise ValueError("cl must be in (0,1)")

    n_obs = int(math.floor(b))
    target = 1.0 - cl

    def f(s: float) -> float:
        return _poisson_cdf(n_obs, b + s) - target

    lo = 0.0
    hi = max(5.0, 5.0 * math.sqrt(b + 1.0) + 5.0)

    # Ensure bracket
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            return hi

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def s95_gaussian_background_dominated(B: float, dB: float = 0.0) -> float:
    """
    Fast rule-of-thumb (one-sided 95%):
        S95 ≈ 1.64 * sqrt(B + dB^2)
    """
    if B < 0 or dB < 0:
        raise ValueError("B and dB must be >= 0")
    return 1.64 * math.sqrt(B + dB * dB)


def s95_auto(B: float, dB: float = 0.0, threshold: float = 10.0) -> Tuple[float, str]:
    """
    Auto-select S95 method:
      - if B < threshold and dB == 0 -> Poisson background-known
      - otherwise -> Gaussian approximation
    Returns: (S95, method_used)
    """
    if B < 0 or dB < 0:
        raise ValueError("B and dB must be >= 0")
    if B < threshold and dB == 0.0:
        return s95_poisson_background_known(B, cl=0.95), "poisson"
    return s95_gaussian_background_dominated(B, dB=dB), "gauss"


def compute_s95(B: float, dB: float = 0.0, method: str = "auto", lowB_threshold: float = 10.0) -> Tuple[float, str]:
    """
    Unified S95 computation.
    method: 'auto' | 'gauss' | 'poisson'
    """
    method = method.lower().strip()
    if method == "gauss":
        return s95_gaussian_background_dominated(B, dB=dB), "gauss"
    if method == "poisson":
        if dB != 0.0:
            raise ValueError("Poisson background-known requires dB=0. Use gauss/auto otherwise.")
        return s95_poisson_background_known(B, cl=0.95), "poisson"
    if method == "auto":
        return s95_auto(B, dB=dB, threshold=lowB_threshold)
    raise ValueError("method must be one of: auto | gauss | poisson")


# =============================================================================
# ε extraction
# =============================================================================

def epsilon95_quadratic(S95: float, S0: float, floor: float = 1e-12) -> float:
    """
    Quadratic scaling:
      S = ε^2 S0 => ε95 = sqrt(S95/S0)
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if S95 <= 0:
        return floor
    return max(math.sqrt(S95 / S0), floor)


def epsilon95_linear(S95: float, S0: float, floor: float = 1e-12) -> float:
    """
    Linear scaling benchmark:
      S = ε S0 => ε95 = S95/S0
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if S95 <= 0:
        return floor
    return max(S95 / S0, floor)


def epsilon_from_mu(mu: float, floor: float = 0.0) -> float:
    """
    If you have a fitted signal strength mu for an ε=1 template in quadratic models:
      mu = ε^2  =>  ε = sqrt(mu)
    """
    if mu <= 0:
        return floor
    return max(math.sqrt(mu), floor)


# =============================================================================
# Benchmarks
# =============================================================================

DEFAULT_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # These are intentionally illustrative. Replace with paper-specific numbers as needed.
    "10 TeV optimistic": {"bkg": 5.0, "S0_per_ab": 50.0, "dB": 0.0},
    "10 TeV realistic":  {"bkg": 20.0, "S0_per_ab": 100.0, "dB": 0.0},
    "3 TeV early":       {"bkg": 10.0, "S0_per_ab": 20.0, "dB": 0.0},

    # Slide-worthy example suggested in-thread:
    "10 TeV slide demo (B=8,S0/ab=80)": {"bkg": 8.0, "S0_per_ab": 80.0, "dB": 0.0},
}


# =============================================================================
# Tables + plots
# =============================================================================

def plot_sensitivity(
    lumi_ab: Sequence[float],
    bkg: float,
    S0_per_ab: float,
    *,
    dB: float = 0.0,
    method: str = "auto",
    lowB_threshold: float = 10.0,
    acceptance: float = 1.0,
    efficiency: float = 1.0,
    title: Optional[str] = None,
    outfile: str = "epsilon_sensitivity.png"
) -> Dict[str, Any]:
    """
    Plot ε95 vs luminosity for quadratic vs linear scaling.
    Returns dict with S95, method_used, arrays, and outfile.
    """
    lumi = np.array(list(lumi_ab), dtype=float)
    if lumi.size == 0:
        raise ValueError("lumi_ab must not be empty")
    if np.any(lumi <= 0):
        raise ValueError("All luminosities must be > 0")
    if bkg < 0 or S0_per_ab <= 0:
        raise ValueError("bkg must be >= 0 and S0_per_ab must be > 0")
    if acceptance < 0 or efficiency < 0:
        raise ValueError("acceptance/efficiency must be >= 0")

    S95, method_used = compute_s95(bkg, dB=dB, method=method, lowB_threshold=lowB_threshold)

    # expected ε=1 yield as function of luminosity
    S0 = S0_per_ab * lumi * acceptance * efficiency

    eps_quad = np.array([epsilon95_quadratic(S95, s0) for s0 in S0], dtype=float)
    eps_lin = np.array([epsilon95_linear(S95, s0) for s0 in S0], dtype=float)

    plt.figure(figsize=(7.2, 5.0))
    plt.loglog(lumi, eps_quad, "o-", label="Quadratic (Majorana/Hyperknot): S=ε² S0")
    plt.loglog(lumi, eps_lin, "s--", label="Linear benchmark: S=ε S0")
    plt.xlabel(r"Integrated luminosity [ab$^{-1}$]")
    plt.ylabel(r"95% CL bound on $\varepsilon$")

    if title is None:
        title = f"ε Sensitivity Scaling ({method_used}; B={bkg:g}, S0/ab={S0_per_ab:g})"
    plt.title(title)

    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

    return {
        "S95": float(S95),
        "method_used": method_used,
        "lumi_ab": lumi.tolist(),
        "eps95_quadratic": eps_quad.tolist(),
        "eps95_linear": eps_lin.tolist(),
        "outfile": outfile,
    }


def benchmark_table(
    benchmarks: Dict[str, Dict[str, float]],
    lumi_ab: Sequence[float],
    *,
    method: str = "auto",
    lowB_threshold: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Return a list of dict rows suitable for printing, JSON, or conversion to LaTeX.
    """
    lumi = list(lumi_ab)
    if len(lumi) == 0:
        raise ValueError("lumi_ab must not be empty")

    rows: List[Dict[str, Any]] = []
    for name, cfg in benchmarks.items():
        bkg = float(cfg.get("bkg", 0.0))
        S0_per_ab = float(cfg.get("S0_per_ab", 0.0))
        dB = float(cfg.get("dB", 0.0))

        if S0_per_ab <= 0:
            raise ValueError(f"Benchmark '{name}' must define S0_per_ab > 0")

        S95, method_used = compute_s95(bkg, dB=dB, method=method, lowB_threshold=lowB_threshold)

        for L in lumi:
            Lf = float(L)
            if Lf <= 0:
                raise ValueError("All luminosities must be > 0")
            S0 = S0_per_ab * Lf
            eps_q = epsilon95_quadratic(S95, S0)
            eps_l = epsilon95_linear(S95, S0)
            rows.append({
                "benchmark": name,
                "L_ab": Lf,
                "B": bkg,
                "dB": dB,
                "S0_per_ab": S0_per_ab,
                "S95": float(S95),
                "method_used": method_used,
                "eps95_quadratic": float(eps_q),
                "eps95_linear": float(eps_l),
            })
    return rows


def print_table(rows: List[Dict[str, Any]], *, file=sys.stdout) -> None:
    """
    Human-friendly table print.
    """
    if not rows:
        print("(no rows)", file=file)
        return

    # Column widths
    bench_w = max(len(str(r["benchmark"])) for r in rows)
    header = (
        f"{'benchmark'.ljust(bench_w)}  "
        f"{'L[ab^-1]':>8}  {'B':>6}  {'dB':>6}  {'S0/ab':>8}  "
        f"{'S95':>8}  {'eps95_quad':>10}  {'eps95_lin':>10}  {'method':>7}"
    )
    print(header, file=file)
    print("-" * len(header), file=file)
    for r in rows:
        print(
            f"{str(r['benchmark']).ljust(bench_w)}  "
            f"{r['L_ab']:8.1f}  {r['B']:6.1f}  {r['dB']:6.1f}  {r['S0_per_ab']:8.1f}  "
            f"{r['S95']:8.2f}  {r['eps95_quadratic']:10.4g}  {r['eps95_linear']:10.4g}  {str(r['method_used']):>7}",
            file=file
        )


def latex_table(rows: List[Dict[str, Any]]) -> str:
    """
    Emit a simple LaTeX tabular. (No external LaTeX deps.)
    """
    if not rows:
        return "% (no rows)\n"

    # group by benchmark
    by_bench: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_bench.setdefault(str(r["benchmark"]), []).append(r)

    # sort within each bench by L
    for k in by_bench:
        by_bench[k] = sorted(by_bench[k], key=lambda x: float(x["L_ab"]))

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Benchmark & $L$ [ab$^{-1}$] & $B$ & $S_{95}$ & $S_0$/ab & $\varepsilon_{95}^{(\mathrm{quad})}$ & $\varepsilon_{95}^{(\mathrm{lin})}$ \\")
    lines.append(r"\hline")
    for bench, brs in by_bench.items():
        for r in brs:
            lines.append(
                f"{bench} & {r['L_ab']:.1f} & {r['B']:.1f} & {r['S95']:.2f} & {r['S0_per_ab']:.1f} & {r['eps95_quadratic']:.4g} & {r['eps95_linear']:.4g} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Projected 95\% CL bounds on $\varepsilon$ under quadratic vs linear scaling.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================

def _parse_lumi(vals: List[str]) -> List[float]:
    if not vals:
        raise ValueError("Must provide at least one luminosity value.")
    out = []
    for v in vals:
        out.append(float(v))
    return out


def cli_plot(args: argparse.Namespace) -> int:
    lumi = _parse_lumi(args.lumi)
    result = plot_sensitivity(
        lumi_ab=lumi,
        bkg=args.bkg,
        S0_per_ab=args.S0_per_ab,
        dB=args.dB,
        method=args.method,
        lowB_threshold=args.lowB_threshold,
        acceptance=args.acceptance,
        efficiency=args.efficiency,
        title=args.title,
        outfile=args.outfile,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Wrote: {result['outfile']}")
        print(f"S95={result['S95']:.3f} (method={result['method_used']})")
        for L, eq, el in zip(result["lumi_ab"], result["eps95_quadratic"], result["eps95_linear"]):
            print(f"  L={L:g} ab^-1  eps95_quad={eq:.6g}  eps95_lin={el:.6g}")
    return 0


def cli_table(args: argparse.Namespace) -> int:
    lumi = _parse_lumi(args.lumi)

    if args.preset == "default":
        benchmarks = DEFAULT_BENCHMARKS
    else:
        # Load JSON from file for custom benchmarks
        with open(args.preset, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)
        if not isinstance(benchmarks, dict):
            raise ValueError("Benchmark JSON must be an object/dict of name -> config")

    rows = benchmark_table(
        benchmarks=benchmarks,
        lumi_ab=lumi,
        method=args.method,
        lowB_threshold=args.lowB_threshold
    )

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    elif args.format == "latex":
        print(latex_table(rows))
    else:
        print_table(rows)
    return 0


def cli_eps(args: argparse.Namespace) -> int:
    """
    Utility: compute ε from mu or from (S95,S0).
    """
    if args.from_mu is not None:
        eps = epsilon_from_mu(args.from_mu)
        print(f"epsilon = sqrt(mu) = {eps:.8g}")
        return 0

    if args.S95 is None or args.S0 is None:
        raise ValueError("Must provide either --from-mu or both --S95 and --S0")

    if args.scaling == "quad":
        eps = epsilon95_quadratic(args.S95, args.S0)
        print(f"epsilon95 (quadratic) = sqrt(S95/S0) = {eps:.8g}")
    else:
        eps = epsilon95_linear(args.S95, args.S0)
        print(f"epsilon95 (linear) = S95/S0 = {eps:.8g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NeutrinoColliderMath: ε-extraction under quadratic vs linear scaling for ΔL=2 collider processes."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # plot
    sp = sub.add_parser("plot", help="Generate ε95 sensitivity plot vs luminosity.")
    sp.add_argument("--lumi", nargs="+", required=True, help="Luminosities in ab^-1 (space-separated).")
    sp.add_argument("--bkg", type=float, required=True, help="Expected background event count B.")
    sp.add_argument("--S0-per-ab", type=float, required=True, help="Expected ε=1 signal yield per ab^-1.")
    sp.add_argument("--dB", type=float, default=0.0, help="Background uncertainty (Gaussian method).")
    sp.add_argument("--method", choices=["auto", "gauss", "poisson"], default="auto", help="S95 method.")
    sp.add_argument("--lowB-threshold", type=float, default=10.0, help="Threshold for auto Poisson vs Gaussian.")
    sp.add_argument("--acceptance", type=float, default=1.0, help="Acceptance factor (optional).")
    sp.add_argument("--efficiency", type=float, default=1.0, help="Efficiency factor (optional).")
    sp.add_argument("--title", type=str, default=None, help="Plot title (optional).")
    sp.add_argument("--outfile", type=str, default="epsilon_sensitivity.png", help="Output image file.")
    sp.add_argument("--json", action="store_true", help="Print JSON result.")
    sp.set_defaults(func=cli_plot)

    # table
    st = sub.add_parser("table", help="Print a benchmark table of ε95 for multiple luminosities.")
    st.add_argument("--preset", type=str, default="default",
                    help="Benchmark preset: 'default' or path to JSON file containing benchmarks dict.")
    st.add_argument("--lumi", nargs="+", required=True, help="Luminosities in ab^-1 (space-separated).")
    st.add_argument("--method", choices=["auto", "gauss", "poisson"], default="auto", help="S95 method.")
    st.add_argument("--lowB-threshold", type=float, default=10.0, help="Threshold for auto Poisson vs Gaussian.")
    st.add_argument("--format", choices=["text", "json", "latex"], default="text", help="Output format.")
    st.set_defaults(func=cli_table)

    # eps utility
    se = sub.add_parser("eps", help="Compute ε from mu or from (S95,S0).")
    se.add_argument("--from-mu", type=float, default=None, help="Compute ε = sqrt(mu).")
    se.add_argument("--S95", type=float, default=None, help="95% CL signal event limit S95.")
    se.add_argument("--S0", type=float, default=None, help="Expected ε=1 signal yield S0.")
    se.add_argument("--scaling", choices=["quad", "lin"], default="quad", help="Scaling for ε from S95/S0.")
    se.set_defaults(func=cli_eps)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
