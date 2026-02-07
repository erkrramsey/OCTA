#!/usr/bin/env python3
# ============================================================
# QC Geometric Energy Cell — Quasicrystal Geometry Atlas + Multi-Domain Mapping
# v0.28.0 (single-file, Colab/CLI turnkey)
#
# FIXED / UPGRADED:
# - PENW collapse fixed (robust fallback + adaptive relaxation + adaptive r_min)
# - Resistor singular fixed (electrode-connected subgraph + optional shunt regularization)
# - PENCP edges=0 fixed (auto edges: shells -> knn fallback)
# - AB edges improved (multi-shell lengths 1 and sqrt(2), with degree controls)
# - Hard numbers report (connectivity, components, spectra, Geff, energy feasibility)
# - One ZIP bundle + optional Colab auto-download
#
# Domains (per job):
# - Photonics: tight-binding on weighted adjacency
# - Phonons: Laplacian eigenmodes
# - Resistor network: effective conductance left->right
#
# Output:
# - atlas renders (patch/deploy/heatmaps) with coordinate grid
# - per-domain outputs (plots + JSON)
# - energy sweeps + feasibility table
# - report.json + report.md
# - one ZIP bundle
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional SciPy (recommended)
_SCIPY_OK = False
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# ----------------------------
# Utilities
# ----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def mkdirp(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

def save_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def _strip_ipykernel_argv(argv: List[str]) -> List[str]:
    out = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "-f":
            skip_next = True
            continue
        if a.startswith("--f="):
            continue
        out.append(a)
    return out

def robust_percentiles(x: np.ndarray, ps=(10, 50, 90)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    vals = np.percentile(x, list(ps))
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}

def estimate_nn_median(P: np.ndarray, sample: int = 256, seed: int = 0) -> float:
    n = P.shape[0]
    if n <= 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    mins = []
    for i in idx:
        d2 = np.sum((P - P[i]) ** 2, axis=1)
        d2[i] = np.inf
        mins.append(math.sqrt(float(np.min(d2))))
    med = float(np.median(np.array(mins)))
    return med if med > 1e-12 else 1.0

def _nearest_neighbor_dist(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    if n <= 2:
        return np.array([0.0], dtype=float)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        d2 = np.sum((P - P[i])**2, axis=1)
        d2[i] = np.inf
        out[i] = math.sqrt(float(np.min(d2)))
    return out

def _extent_from_points(P: np.ndarray, pad: float = 1.0) -> Tuple[float,float,float,float]:
    if P.shape[0] == 0:
        return (-10, 10, -10, 10)
    xmin, xmax = float(np.min(P[:,0])), float(np.max(P[:,0]))
    ymin, ymax = float(np.min(P[:,1])), float(np.max(P[:,1]))
    if abs(xmax - xmin) < 1e-9:
        xmin -= 1.0; xmax += 1.0
    if abs(ymax - ymin) < 1e-9:
        ymin -= 1.0; ymax += 1.0
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

def grid_hash(P: np.ndarray, cell: float) -> Dict[Tuple[int, int], List[int]]:
    H: Dict[Tuple[int, int], List[int]] = {}
    inv = 1.0 / max(cell, 1e-12)
    for i, (x, y) in enumerate(P):
        cx = int(math.floor(x * inv))
        cy = int(math.floor(y * inv))
        H.setdefault((cx, cy), []).append(i)
    return H

def neighbor_candidates(H: Dict[Tuple[int, int], List[int]], cx: int, cy: int, ring: int = 1) -> List[int]:
    out = []
    for dx in range(-ring, ring+1):
        for dy in range(-ring, ring+1):
            out.extend(H.get((cx + dx, cy + dy), []))
    return out


# ----------------------------
# Graph stats / components
# ----------------------------

def build_adj_list(n: int, E: np.ndarray) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for a, b in E:
        a = int(a); b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    return adj

def connected_components(n: int, E: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    """Returns comp_id per node and list of components (list of node indices)."""
    if n == 0:
        return np.zeros((0,), dtype=int), []
    adj = build_adj_list(n, E)
    comp_id = -np.ones(n, dtype=int)
    comps: List[List[int]] = []
    cid = 0
    for i in range(n):
        if comp_id[i] != -1:
            continue
        # BFS
        q = [i]
        comp_id[i] = cid
        nodes = [i]
        for v in q:
            for u in adj[v]:
                if comp_id[u] == -1:
                    comp_id[u] = cid
                    q.append(u)
                    nodes.append(u)
        comps.append(nodes)
        cid += 1
    return comp_id, comps

def component_stats(n: int, E: np.ndarray) -> dict:
    comp_id, comps = connected_components(n, E)
    sizes = [len(c) for c in comps]
    sizes_sorted = sorted(sizes, reverse=True)
    largest = sizes_sorted[0] if sizes_sorted else 0
    return {
        "n": int(n),
        "m": int(E.shape[0]),
        "components": int(len(comps)),
        "largest_size": int(largest),
        "largest_frac": float(largest / max(n, 1)),
        "size_top5": sizes_sorted[:5],
    }

def electrode_connected_mask(P: np.ndarray, E: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Mask of nodes connected to either electrode through graph connectivity."""
    n = P.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if E.shape[0] == 0:
        return np.zeros((n,), dtype=bool)
    adj = build_adj_list(n, E)
    seeds = list(np.unique(np.concatenate([left, right]).astype(int)))
    mask = np.zeros(n, dtype=bool)
    q = []
    for s in seeds:
        if 0 <= s < n:
            mask[s] = True
            q.append(int(s))
    for v in q:
        for u in adj[v]:
            if not mask[u]:
                mask[u] = True
                q.append(u)
    return mask


# ----------------------------
# Specs
# ----------------------------

@dataclass
class RenderSpec:
    dark: bool = True
    figsize: Tuple[float, float] = (8.2, 8.2)
    dpi: int = 170
    show_axes: bool = True
    show_grid: bool = True
    grid_major: float = 5.0
    grid_minor: float = 1.0
    grid_alpha_major: float = 0.45
    grid_alpha_minor: float = 0.16
    point_size: float = 7.0
    point_alpha: float = 0.75
    edge_lw: float = 1.0
    edge_alpha: float = 0.28
    sel_edge_lw: float = 3.0
    sel_edge_alpha: float = 0.95
    title_size: int = 18
    equal_aspect: bool = True

@dataclass
class DeploySpec:
    deploy_frac: float = 0.08
    nbins: int = 12
    phase: float = 0.6
    jitter: float = 0.10
    mode: str = "topk"          # "topk" or "threshold"
    threshold: float = 0.75     # used for "threshold"
    field_gain: float = 0.6     # HYBRID mixing

@dataclass
class PlacementSpec:
    bbox_R: float = 22.0

@dataclass
class ABSpec:
    window_w: float = 1.96
    normalize_shortest_nn: bool = True
    edge_tol: float = 0.018
    max_deg: int = 6
    R_lattice: int = 14

@dataclass
class PenWaveSpec:
    L: float = 18.0
    dx: float = 0.25
    k: float = 1.0
    seed: int = 9
    peak_quantile: float = 0.83
    r_min: float = 0.35
    topK: int = 2200
    min_peaks: int = 1150
    use_local_maxima: bool = True
    fallback_full_field: bool = True
    margin_cells: int = 2
    max_deg: int = 8
    edge_policy: str = "knn"       # "knn" (robust) or "shell"
    knn_k: int = 6
    knn_rmax: float = 1.8
    edge_tol: float = 0.03         # used for shell

@dataclass
class PenCPSpec:
    R_lattice: int = 6
    window_w: float = 2.2
    bbox_R: float = 22.0
    max_deg: int = 8
    edge_policy: str = "knn"
    knn_k: int = 6
    knn_rmax: float = 1.8
    edge_tol: float = 0.03
    seed: int = 4

@dataclass
class SquareSpec:
    spacing: float = 1.0
    half_extent: int = 18

@dataclass
class HybridSpec:
    base_name: str = "AB_n4_4"
    field_map_scale: float = 0.78
    pen_field_L: float = 18.0
    pen_field_dx: float = 0.25
    pen_field_k: float = 1.0
    pen_field_seed: int = 9

@dataclass
class EnergySpec:
    sim_s: float = 30.0
    dt_s: float = 0.001
    C_store_uF: float = 330.0
    V_init: float = 3.0
    V_dump_lo: float = 3.2
    V_dump_hi: float = 5.2
    eta_dump: float = 0.85
    I_leak_uA: float = 2.0
    f_edge_hz: float = 0.9
    q_per_event_nC: float = 1.4

@dataclass
class SweepSpec:
    C_store_uF_list: List[float] = field(default_factory=lambda: [10, 22, 47, 100, 220, 330, 470, 1000])
    V_dump_hi_list: List[float] = field(default_factory=lambda: [3.6, 3.8, 4.2, 4.6, 5.2, 6.0, 7.0])
    I_leak_uA_list: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.5, 1.5, 2.0, 5, 10, 15])
    edges_list: List[int] = field(default_factory=lambda: [200, 400, 800, 1200, 1600, 2200, 3000, 6000])

# Domain mapping specs
@dataclass
class PhotonicsSpec:
    w0: float = 200.0
    kappa: float = 8.0
    base_coupling: float = 1.0
    sel_boost: float = 2.2
    boundary_loss_gamma: float = 0.015
    modes_to_render: int = 4
    spectrum_bins: int = 60
    max_dense_n: int = 1100
    k_sparse: int = 220

@dataclass
class PhononSpec:
    mass: float = 1.0
    k_spring: float = 1.0
    base_stiffness: float = 1.0
    sel_boost: float = 2.2
    modes_to_render: int = 4
    spectrum_bins: int = 60
    max_dense_n: int = 1100
    k_sparse: int = 220

@dataclass
class ResistorSpec:
    base_g: float = 1.0
    sel_boost: float = 2.2
    electrode_frac: float = 0.06
    V_left: float = 1.0
    V_right: float = 0.0
    # robustness:
    use_electrode_component_only: bool = True
    shunt_eps: float = 0.0  # if >0, add eps*I to L_ff (regularizes)

@dataclass
class DomainSuiteSpec:
    run_photonics: bool = True
    run_phonons: bool = True
    run_resistor: bool = True
    parallel: bool = True
    phot: PhotonicsSpec = field(default_factory=PhotonicsSpec)
    phon: PhononSpec = field(default_factory=PhononSpec)
    res: ResistorSpec = field(default_factory=ResistorSpec)

@dataclass
class GeometryJob:
    name: str
    kind: str  # "AB", "PENW", "PENCP", "SQUARE", "HYBRID"
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    deploy: DeploySpec = field(default_factory=DeploySpec)
    ab: Optional[ABSpec] = None
    penw: Optional[PenWaveSpec] = None
    pencp: Optional[PenCPSpec] = None
    square: Optional[SquareSpec] = None
    hybrid: Optional[HybridSpec] = None


# ----------------------------
# Geometry: Ammann–Beenker cut-and-project
# ----------------------------

def _inside_regular_octagon(u: float, v: float, w: float) -> bool:
    au = abs(u)
    av = abs(v)
    return max(au, av, (au + av) / math.sqrt(2.0)) <= w

def make_ab_patch(spec: ABSpec, bbox_R: float, seed: int = 0) -> Tuple[np.ndarray, dict]:
    # Ammann–Beenker with 4D projection
    thetas = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
    phys = np.array([[math.cos(t) for t in thetas],
                     [math.sin(t) for t in thetas]], dtype=float)  # 2x4
    intr = np.array([[math.cos(t + math.pi/2) for t in thetas],
                     [math.sin(t + math.pi/2) for t in thetas]], dtype=float)  # 2x4

    R = int(spec.R_lattice)
    pts = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            for c in range(-R, R + 1):
                for d in range(-R, R + 1):
                    n = np.array([a, b, c, d], dtype=float)
                    x = phys @ n
                    y = intr @ n
                    if _inside_regular_octagon(float(y[0]), float(y[1]), float(spec.window_w)):
                        if (x[0]*x[0] + x[1]*x[1]) <= (bbox_R*bbox_R*4.0):
                            pts.append([float(x[0]), float(x[1])])

    Pxy = np.array(pts, dtype=float)
    if Pxy.shape[0] == 0:
        return np.zeros((0,2), dtype=float), {"ab": {"empty": True}}

    nn_med = estimate_nn_median(Pxy, sample=512, seed=seed)
    if spec.normalize_shortest_nn and nn_med > 1e-12:
        Pxy = Pxy / nn_med

    keep = np.sum(Pxy**2, axis=1) <= (bbox_R*bbox_R)
    Pxy = Pxy[keep]

    diag = {
        "nn": {
            "median": float(estimate_nn_median(Pxy, seed=seed)),
            **robust_percentiles(_nearest_neighbor_dist(Pxy), (10, 50, 90))
        },
        "ab": {
            "window_w": float(spec.window_w),
            "normalize_shortest_nn": bool(spec.normalize_shortest_nn),
            "edge_tol": float(spec.edge_tol),
            "max_deg": int(spec.max_deg),
            "R_lattice": int(spec.R_lattice),
            "bbox_R": float(bbox_R),
        }
    }
    return Pxy, diag


# ----------------------------
# Geometry: Penrose-like wave-field peaks patch (ROBUST)
# ----------------------------

def local_maxima_2d(A: np.ndarray) -> np.ndarray:
    h, w = A.shape
    if h < 3 or w < 3:
        return np.zeros((0,2), dtype=np.int32)
    C = A[1:-1, 1:-1]
    neigh = [
        A[:-2, 1:-1], A[2:, 1:-1], A[1:-1, :-2], A[1:-1, 2:],
        A[:-2, :-2], A[:-2, 2:], A[2:, :-2], A[2:, 2:]
    ]
    M = np.ones_like(C, dtype=bool)
    for N in neigh:
        M &= (C > N)
    ys, xs = np.where(M)
    return np.stack([ys + 1, xs + 1], axis=1).astype(np.int32)

def greedy_rmin_select(cand: np.ndarray, r_min: float, want: int) -> np.ndarray:
    if cand.shape[0] == 0:
        return np.zeros((0,2), dtype=float)
    r2 = r_min * r_min
    cell = max(r_min, 1e-6)
    inv = 1.0 / cell
    H: Dict[Tuple[int,int], List[int]] = {}
    coords: List[Tuple[float,float]] = []

    for x, y in cand:
        x = float(x); y = float(y)
        cx = int(math.floor(x * inv))
        cy = int(math.floor(y * inv))
        ok = True
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for idx in H.get((cx+dx, cy+dy), []):
                    xx, yy = coords[idx]
                    ddx = x - xx
                    ddy = y - yy
                    if ddx*ddx + ddy*ddy < r2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            coords.append((x,y))
            H.setdefault((cx,cy), []).append(len(coords)-1)
            if len(coords) >= want:
                break

    return np.array(coords, dtype=float)

def make_pen_wave_points(spec: PenWaveSpec, seed_override: Optional[int] = None) -> Tuple[np.ndarray, dict, dict]:
    seed = int(spec.seed if seed_override is None else seed_override)
    rng = np.random.default_rng(seed)
    L = float(spec.L)
    dx = float(spec.dx)
    k = float(spec.k)

    xs = np.arange(-L, L + 1e-12, dx)
    ys = np.arange(-L, L + 1e-12, dx)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    angs = [2.0 * math.pi * i / 5.0 for i in range(5)]
    phases = rng.uniform(0, 2.0 * math.pi, size=5)

    F = np.zeros_like(X, dtype=float)
    for a, ph in zip(angs, phases):
        ux = math.cos(a); uy = math.sin(a)
        F += np.cos(k * (ux * X + uy * Y) + ph)

    m = int(spec.margin_cells)
    if m > 0 and X.shape[0] > 2*m and X.shape[1] > 2*m:
        Fc = F[m:-m, m:-m]
        Xc = X[m:-m, m:-m]
        Yc = Y[m:-m, m:-m]
    else:
        Fc, Xc, Yc = F, X, Y

    diag = {
        "grid": {"nx": int(X.shape[1]), "ny": int(X.shape[0]), "dx": float(dx), "L": float(L)},
        "seed": seed,
        "use_local_maxima": bool(spec.use_local_maxima),
        "fallback_full_field": bool(spec.fallback_full_field),
        "peak_quantile_start": float(spec.peak_quantile),
        "r_min_start": float(spec.r_min),
        "local_maxima": 0,
        "cand_after_threshold": 0,
        "points_out": 0,
        "relax_steps": [],
    }

    # Candidate generation
    peaks = np.zeros((0,2), dtype=np.int32)
    vals = np.zeros((0,), dtype=float)

    if spec.use_local_maxima:
        peaks = local_maxima_2d(Fc)
        diag["local_maxima"] = int(peaks.shape[0])
        if peaks.shape[0] > 0:
            vals = Fc[peaks[:, 0], peaks[:, 1]]
            # initial threshold on maxima values
            thr = np.quantile(vals, float(spec.peak_quantile)) if vals.size else np.inf
            keep = vals >= thr
            peaks = peaks[keep]
            vals = vals[keep]

        # Robust fallback if too few maxima survive
        if spec.fallback_full_field and peaks.shape[0] < max(64, int(spec.min_peaks // 4)):
            vals_full = Fc.ravel()
            thr = np.quantile(vals_full, float(spec.peak_quantile))
            keep = vals_full >= thr
            idx = np.where(keep)[0]
            peaks = np.stack(np.unravel_index(idx, Fc.shape), axis=1).astype(np.int32)
            vals = Fc[peaks[:, 0], peaks[:, 1]]

    else:
        vals_full = Fc.ravel()
        thr = np.quantile(vals_full, float(spec.peak_quantile))
        keep = vals_full >= thr
        idx = np.where(keep)[0]
        peaks = np.stack(np.unravel_index(idx, Fc.shape), axis=1).astype(np.int32)
        vals = Fc[peaks[:, 0], peaks[:, 1]]

    # Adaptive relaxation if still too few candidates
    q = float(spec.peak_quantile)
    while peaks.shape[0] < int(spec.min_peaks) and q > 0.55:
        q = max(0.55, q - 0.02)
        if vals.size == 0:
            break
        thr = np.quantile(vals, q)
        keep = vals >= thr
        peaks = peaks[keep]
        vals = vals[keep]
        diag["relax_steps"].append({"q": float(q), "cand": int(peaks.shape[0])})
        if peaks.shape[0] >= int(spec.min_peaks):
            break

    # Sort candidates strongest first
    if vals.size > 0:
        order = np.argsort(-vals)
        peaks = peaks[order]
        vals = vals[order]

    # Convert peaks -> coordinates
    if peaks.shape[0] == 0:
        return np.zeros((0,2), dtype=float), {"pen": {"empty": True}}, diag

    cand = np.stack([Xc[peaks[:, 0], peaks[:, 1]], Yc[peaks[:, 0], peaks[:, 1]]], axis=1)
    if cand.shape[0] > int(spec.topK):
        cand = cand[:int(spec.topK)]
    diag["cand_after_threshold"] = int(cand.shape[0])

    # Adaptive r_min: if we fail to hit target, relax spacing
    want = int(spec.min_peaks)
    r_min = float(spec.r_min)
    P = greedy_rmin_select(cand, r_min=r_min, want=want)
    attempts = 0
    while P.shape[0] < max(64, int(0.85 * want)) and attempts < 6:
        r_min = max(0.12, r_min * 0.88)
        P = greedy_rmin_select(cand, r_min=r_min, want=want)
        attempts += 1
        diag["relax_steps"].append({"r_min": float(r_min), "selected": int(P.shape[0])})

    # Normalize by NN median
    nn_med = estimate_nn_median(P, seed=seed)
    if nn_med > 1e-12:
        P = P / nn_med

    stats = {
        "L": float(spec.L),
        "dx": float(spec.dx),
        "k": float(spec.k),
        "seed": int(seed),
        "peak_quantile": float(spec.peak_quantile),
        "r_min_final": float(r_min),
        "topK": int(spec.topK),
        "min_peaks": int(spec.min_peaks),
        "edge_policy": str(spec.edge_policy),
        "knn_k": int(spec.knn_k),
        "knn_rmax": float(spec.knn_rmax),
        "max_deg": int(spec.max_deg),
        "edge_tol": float(spec.edge_tol),
    }
    diag["points_out"] = int(P.shape[0])
    return P, stats, diag


# ----------------------------
# Geometry: Penrose CP variant (5D -> 2D), spherical internal window approx
# ----------------------------

def make_pen_cp_points(spec: PenCPSpec) -> Tuple[np.ndarray, dict]:
    thetas = [2.0 * math.pi * i / 5.0 for i in range(5)]
    u = np.array([math.cos(t) for t in thetas], dtype=float)
    v = np.array([math.sin(t) for t in thetas], dtype=float)

    b1 = u / (np.linalg.norm(u) + 1e-12)
    v2 = v - np.dot(v, b1) * b1
    b2 = v2 / (np.linalg.norm(v2) + 1e-12)

    Bint = []
    for i in range(5):
        e = np.zeros(5, dtype=float)
        e[i] = 1.0
        w = e - np.dot(e, b1) * b1 - np.dot(e, b2) * b2
        for q in Bint:
            w = w - np.dot(w, q) * q
        nrm = np.linalg.norm(w)
        if nrm > 1e-9:
            Bint.append(w / nrm)
        if len(Bint) >= 3:
            break
    Bint = np.stack(Bint, axis=0)  # 3x5

    R = int(spec.R_lattice)
    pts = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            for c in range(-R, R+1):
                for d in range(-R, R+1):
                    for e in range(-R, R+1):
                        n = np.array([a,b,c,d,e], dtype=float)
                        x = np.array([np.dot(b1, n), np.dot(b2, n)], dtype=float)
                        y = Bint @ n
                        if float(np.linalg.norm(y)) <= float(spec.window_w):
                            if float(x[0]*x[0] + x[1]*x[1]) <= float(spec.bbox_R*spec.bbox_R*4.0):
                                pts.append([float(x[0]), float(x[1])])

    Pxy = np.array(pts, dtype=float)
    if Pxy.shape[0] == 0:
        return np.zeros((0,2), dtype=float), {"pen_cp": {"empty": True}}

    nn_med = estimate_nn_median(Pxy, seed=int(spec.seed))
    if nn_med > 1e-12:
        Pxy = Pxy / nn_med

    keep = np.sum(Pxy**2, axis=1) <= (float(spec.bbox_R) * float(spec.bbox_R))
    Pxy = Pxy[keep]

    diag = {
        "nn": {
            "median": float(estimate_nn_median(Pxy, seed=int(spec.seed))),
            **robust_percentiles(_nearest_neighbor_dist(Pxy), (10,50,90))
        },
        "pen_cp": {
            "R_lattice": int(spec.R_lattice),
            "window_w": float(spec.window_w),
            "bbox_R": float(spec.bbox_R),
            "max_deg": int(spec.max_deg),
            "edge_policy": str(spec.edge_policy),
            "knn_k": int(spec.knn_k),
            "knn_rmax": float(spec.knn_rmax),
            "edge_tol": float(spec.edge_tol),
            "seed": int(spec.seed),
            "window_kind": "spherical_internal_approx"
        }
    }
    return Pxy, diag


# ----------------------------
# Square lattice
# ----------------------------

def make_square(spec: SquareSpec) -> Tuple[np.ndarray, np.ndarray, dict]:
    s = float(spec.spacing)
    he = int(spec.half_extent)
    xs = np.arange(-he, he+1, dtype=int)
    ys = np.arange(-he, he+1, dtype=int)
    pts = []
    idx_map = {}
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            idx_map[(xi, yi)] = len(pts)
            pts.append([float(x*s), float(y*s)])
    P = np.array(pts, dtype=float)

    edges = []
    w = len(xs)
    h = len(ys)
    for yi in range(h):
        for xi in range(w):
            i = idx_map[(xi, yi)]
            for dx, dy in ((1,0), (0,1)):
                xj = xi + dx
                yj = yi + dy
                if 0 <= xj < w and 0 <= yj < h:
                    j = idx_map[(xj, yj)]
                    edges.append((i,j))
    E = np.array(edges, dtype=np.int32)
    diag = {
        "nn": {"median": 1.0, "p10": 1.0, "p50": 1.0, "p90": 1.0},
        "sq": {"spacing": float(spec.spacing), "half_extent": int(spec.half_extent)}
    }
    return P, E, diag


# ----------------------------
# Edge building (ROBUST)
# ----------------------------

def build_edges_shells(P: np.ndarray, shells: List[float], tol: float, max_deg: int, prefer_short: bool = True) -> np.ndarray:
    """
    Connect edges whose distance is within tol of any shell length.
    Uses grid hashing for candidate pruning, respects max_deg.
    """
    n = P.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int32)

    shells = sorted([float(s) for s in shells])
    r_hi = max(shells) * (1.0 + float(tol))
    cell = max(1.2 * r_hi, 1e-6)
    H = grid_hash(P, cell=cell)
    inv = 1.0 / cell

    deg = np.zeros(n, dtype=np.int32)
    edges: List[Tuple[int, int, float]] = []  # store (i,j,dist) to optionally prioritize

    # Candidate collection
    for i, (x, y) in enumerate(P):
        cx = int(math.floor(x * inv))
        cy = int(math.floor(y * inv))
        cands = neighbor_candidates(H, cx, cy, ring=1)
        for j in cands:
            if j <= i:
                continue
            dx = float(P[j, 0] - x)
            dy = float(P[j, 1] - y)
            d = math.sqrt(dx*dx + dy*dy)
            # shell membership
            ok = False
            for s in shells:
                if abs(d - s) <= (float(tol) * s):
                    ok = True
                    break
            if ok:
                edges.append((i, j, d))

    # Prioritize edges: shorter first helps connectivity and avoids long-only
    if prefer_short:
        edges.sort(key=lambda t: t[2])
    else:
        edges.sort(key=lambda t: -t[2])

    out: List[Tuple[int,int]] = []
    for i, j, _d in edges:
        if deg[i] >= max_deg or deg[j] >= max_deg:
            continue
        out.append((i, j))
        deg[i] += 1
        deg[j] += 1

    return np.array(out, dtype=np.int32)

def build_edges_knn(P: np.ndarray, k: int, r_max: float, max_deg: int, seed: int = 0) -> np.ndarray:
    """
    Approx kNN using grid hashing:
    - for each node, examine neighbors in nearby cells up to a search ring
    - select up to k nearest within r_max
    - enforce max_deg
    """
    n = P.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int32)

    k = int(max(1, k))
    r_max = float(max(1e-6, r_max))
    cell = max(0.9 * r_max, 1e-6)
    inv = 1.0 / cell
    H = grid_hash(P, cell=cell)

    deg = np.zeros(n, dtype=np.int32)
    edges_set = set()

    rng = np.random.default_rng(seed)
    order_nodes = np.arange(n)
    rng.shuffle(order_nodes)

    for i in order_nodes:
        if deg[i] >= max_deg:
            continue
        x, y = float(P[i,0]), float(P[i,1])
        cx = int(math.floor(x * inv))
        cy = int(math.floor(y * inv))

        # expand ring until we have enough candidates or hit a cap
        candidates = []
        for ring in (1, 2, 3):
            cands = neighbor_candidates(H, cx, cy, ring=ring)
            candidates = cands
            if len(candidates) >= 10*k:
                break

        # compute distances
        ds = []
        for j in candidates:
            if j == i:
                continue
            dx = float(P[j,0] - x); dy = float(P[j,1] - y)
            d2 = dx*dx + dy*dy
            if d2 <= r_max*r_max:
                ds.append((d2, j))
        if not ds:
            continue
        ds.sort(key=lambda t: t[0])
        picks = [j for _d2, j in ds[:k]]

        for j in picks:
            if deg[i] >= max_deg or deg[j] >= max_deg:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) in edges_set:
                continue
            edges_set.add((a, b))
            deg[i] += 1
            deg[j] += 1

    edges = np.array(sorted(list(edges_set)), dtype=np.int32)
    return edges

def build_edges_auto(
    P: np.ndarray,
    policy: str,
    *,
    shell_lengths: Optional[List[float]] = None,
    edge_tol: float = 0.02,
    knn_k: int = 6,
    knn_rmax: float = 1.8,
    max_deg: int = 8,
    seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Build edges with robustness:
    - "shell": distance shells (optionally multi-shell)
    - "knn": approximate kNN within rmax
    - automatic fallback: if shell returns too few edges, fallback to knn
    """
    policy = str(policy).lower().strip()
    meta = {"policy": policy}

    if P.shape[0] == 0:
        return np.zeros((0,2), dtype=np.int32), {"policy": policy, "empty": True}

    if policy == "shell":
        if not shell_lengths:
            shell_lengths = [1.0]
        E = build_edges_shells(P, shells=shell_lengths, tol=edge_tol, max_deg=max_deg, prefer_short=True)
        meta.update({"shell_lengths": [float(s) for s in shell_lengths], "edge_tol": float(edge_tol), "max_deg": int(max_deg)})
        # fallback if too sparse
        if E.shape[0] < max(1, int(0.3 * P.shape[0])):
            E2 = build_edges_knn(P, k=knn_k, r_max=knn_rmax, max_deg=max_deg, seed=seed)
            meta["fallback"] = {"to": "knn", "knn_k": int(knn_k), "knn_rmax": float(knn_rmax)}
            if E2.shape[0] > E.shape[0]:
                E = E2
        return E, meta

    if policy == "knn":
        E = build_edges_knn(P, k=knn_k, r_max=knn_rmax, max_deg=max_deg, seed=seed)
        meta.update({"knn_k": int(knn_k), "knn_rmax": float(knn_rmax), "max_deg": int(max_deg)})
        return E, meta

    # unknown policy -> default knn
    E = build_edges_knn(P, k=knn_k, r_max=knn_rmax, max_deg=max_deg, seed=seed)
    meta.update({"policy": "knn_default", "knn_k": int(knn_k), "knn_rmax": float(knn_rmax), "max_deg": int(max_deg)})
    return E, meta


# ----------------------------
# Deployment: motif weighting + selection
# ----------------------------

def motif_weight(theta: np.ndarray, nbins: int, phase: float) -> np.ndarray:
    return 0.5 + 0.5*np.cos(nbins*theta + phase)

def edge_midpoints(P: np.ndarray, E: np.ndarray) -> np.ndarray:
    if E.shape[0] == 0:
        return np.zeros((0,2), dtype=float)
    return 0.5 * (P[E[:,0]] + P[E[:,1]])

def deploy_edges_mask(
    P: np.ndarray,
    E: np.ndarray,
    job_kind: str,
    deploy: DeploySpec,
    field_values_at_mid: Optional[np.ndarray] = None,
    seed: int = 0
) -> Tuple[np.ndarray, dict]:
    m = E.shape[0]
    if m == 0:
        return np.zeros((0,), dtype=bool), {"selected": 0, "total": 0, "deploy_frac": float(deploy.deploy_frac)}

    i = E[:,0]
    j = E[:,1]
    dx = P[j,0] - P[i,0]
    dy = P[j,1] - P[i,1]
    ang = np.mod(np.arctan2(dy, dx), math.pi)

    w_m = motif_weight(ang, nbins=int(deploy.nbins), phase=float(deploy.phase))
    wm = (w_m - float(np.min(w_m))) / (float(np.ptp(w_m)) + 1e-12)  # NumPy 2 safe

    score = wm.copy()
    fv_norm = None

    if (job_kind.upper() == "HYBRID") and (field_values_at_mid is not None) and field_values_at_mid.size == m:
        fv = field_values_at_mid.astype(float)
        fv_norm = (fv - float(np.min(fv))) / (float(np.ptp(fv)) + 1e-12)
        score = (1.0 - float(deploy.field_gain)) * score + float(deploy.field_gain) * fv_norm

    rng = np.random.default_rng(seed)
    if deploy.jitter > 0:
        score = score + float(deploy.jitter) * rng.standard_normal(size=score.shape[0])

    k = max(1, int(round(float(deploy.deploy_frac) * m)))
    sel = np.zeros(m, dtype=bool)
    if deploy.mode == "threshold":
        thr = float(deploy.threshold)
        sel = score >= thr
        if int(np.sum(sel)) < k:
            order = np.argsort(-score)
            sel = np.zeros(m, dtype=bool)
            sel[order[:k]] = True
    else:
        order = np.argsort(-score)
        sel[order[:k]] = True

    sel_idx = np.where(sel)[0]
    diag = {
        "selected": int(np.sum(sel)),
        "total": int(m),
        "deploy_frac": float(np.sum(sel) / max(m,1)),
        "nbins": int(deploy.nbins),
        "phase": float(deploy.phase),
        "mode": str(deploy.mode),
        "threshold": float(deploy.threshold),
        "field_gain": float(deploy.field_gain),
        # hard numbers for energy proxy:
        "motif_selected_mean": float(np.mean(wm[sel_idx])) if sel_idx.size else float("nan"),
        "motif_selected_p50": float(np.median(wm[sel_idx])) if sel_idx.size else float("nan"),
        "score_selected_mean": float(np.mean(score[sel_idx])) if sel_idx.size else float("nan"),
        "score_selected_p50": float(np.median(score[sel_idx])) if sel_idx.size else float("nan"),
    }
    if fv_norm is not None:
        diag["field_selected_mean"] = float(np.mean(fv_norm[sel_idx])) if sel_idx.size else float("nan")
    return sel, diag


# ----------------------------
# Hybrid field sampling
# ----------------------------

def pen_wave_field_grid(L: float, dx: float, k: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = np.arange(-L, L + 1e-12, dx)
    ys = np.arange(-L, L + 1e-12, dx)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    angs = [2.0 * math.pi * i / 5.0 for i in range(5)]
    phases = rng.uniform(0, 2.0 * math.pi, size=5)
    F = np.zeros_like(X, dtype=float)
    for a, ph in zip(angs, phases):
        ux = math.cos(a); uy = math.sin(a)
        F += np.cos(k * (ux * X + uy * Y) + ph)
    return xs, ys, F

def sample_field_at_points(xs: np.ndarray, ys: np.ndarray, F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    x0 = float(xs[0]); y0 = float(ys[0])
    xi = np.clip(np.round((Q[:,0] - x0) / dx).astype(int), 0, xs.size-1)
    yi = np.clip(np.round((Q[:,1] - y0) / dy).astype(int), 0, ys.size-1)
    return F[yi, xi].astype(float)


# ----------------------------
# Heatmaps
# ----------------------------

def edge_heatmaps(P: np.ndarray, E: np.ndarray, sel: np.ndarray, nbins: int = 70) -> Tuple[np.ndarray, np.ndarray, Tuple[float,float,float,float]]:
    if P.shape[0] == 0:
        Hc = np.zeros((nbins, nbins), dtype=float)
        Hw = np.zeros((nbins, nbins), dtype=float)
        return Hw, Hc, (-1,1,-1,1)

    xmin, xmax, ymin, ymax = _extent_from_points(P, pad=0.0)
    mids = edge_midpoints(P, E)
    if mids.shape[0] == 0:
        Hc = np.zeros((nbins, nbins), dtype=float)
        Hw = np.zeros((nbins, nbins), dtype=float)
        return Hw, Hc, (xmin,xmax,ymin,ymax)

    i = E[:,0]; j = E[:,1]
    dx = P[j,0] - P[i,0]
    dy = P[j,1] - P[i,1]
    ang = np.mod(np.arctan2(dy, dx), math.pi)
    w = motif_weight(ang, nbins=12, phase=0.6)

    bx = np.linspace(xmin, xmax, nbins+1)
    by = np.linspace(ymin, ymax, nbins+1)
    Hc = np.zeros((nbins, nbins), dtype=float)
    Hw = np.zeros((nbins, nbins), dtype=float)

    use = sel if (sel is not None and sel.shape[0] == E.shape[0]) else np.ones(E.shape[0], dtype=bool)
    idx = np.where(use)[0]
    for t in idx:
        x, y = float(mids[t,0]), float(mids[t,1])
        ix = int(np.searchsorted(bx, x) - 1)
        iy = int(np.searchsorted(by, y) - 1)
        if 0 <= ix < nbins and 0 <= iy < nbins:
            Hc[iy, ix] += 1.0
            Hw[iy, ix] += float(w[t])

    return Hw, Hc, (xmin,xmax,ymin,ymax)


# ----------------------------
# Rendering helpers (grid + axes)
# ----------------------------

def _apply_dark(ax, dark: bool):
    ax.set_facecolor("black" if dark else "white")

def _draw_axes_grid(ax, extent, render: RenderSpec):
    if not render.show_axes and not render.show_grid:
        ax.set_axis_off()
        return

    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if render.equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if not render.show_axes:
        ax.set_axis_off()
        return

    tc = ("white" if render.dark else "black")
    ax.tick_params(colors=tc, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(tc)

    if render.show_grid:
        # Minor grid
        step = float(render.grid_minor)
        if step > 0:
            gx0 = math.floor(xmin/step)*step
            gx1 = math.ceil(xmax/step)*step
            gy0 = math.floor(ymin/step)*step
            gy1 = math.ceil(ymax/step)*step
            xs = np.arange(gx0, gx1+1e-9, step)
            ys = np.arange(gy0, gy1+1e-9, step)
            for x in xs:
                ax.axvline(x, color=tc, alpha=float(render.grid_alpha_minor), lw=0.8)
            for y in ys:
                ax.axhline(y, color=tc, alpha=float(render.grid_alpha_minor), lw=0.8)

        # Major grid
        step = float(render.grid_major)
        if step > 0:
            gx0 = math.floor(xmin/step)*step
            gx1 = math.ceil(xmax/step)*step
            gy0 = math.floor(ymin/step)*step
            gy1 = math.ceil(ymax/step)*step
            xs = np.arange(gx0, gx1+1e-9, step)
            ys = np.arange(gy0, gy1+1e-9, step)
            for x in xs:
                ax.axvline(x, color=tc, alpha=float(render.grid_alpha_major), lw=1.2)
            for y in ys:
                ax.axhline(y, color=tc, alpha=float(render.grid_alpha_major), lw=1.2)

    ax.axvline(0.0, color=("white" if render.dark else "black"), alpha=0.85, lw=1.6)
    ax.axhline(0.0, color=("white" if render.dark else "black"), alpha=0.85, lw=1.6)

def render_patch(P: np.ndarray, E: np.ndarray, title: str, out_png: str, render: RenderSpec):
    fig, ax = plt.subplots(figsize=render.figsize, dpi=render.dpi)
    _apply_dark(ax, render.dark)

    if E.shape[0] > 0:
        i = E[:,0]; j = E[:,1]
        for a, b in zip(i, j):
            ax.plot([P[a,0], P[b,0]], [P[a,1], P[b,1]],
                    color="white", alpha=float(render.edge_alpha), lw=float(render.edge_lw))

    if P.shape[0] > 0:
        ax.scatter(P[:,0], P[:,1], s=float(render.point_size),
                   c="#66ccff", alpha=float(render.point_alpha), linewidths=0)

    extent = _extent_from_points(P)
    _draw_axes_grid(ax, extent, render)
    ax.set_title(title, color=("white" if render.dark else "black"), fontsize=render.title_size, pad=12)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=("black" if render.dark else "white"))
    plt.close(fig)

def render_deploy(P: np.ndarray, E: np.ndarray, sel: np.ndarray, title: str, out_png: str, render: RenderSpec):
    fig, ax = plt.subplots(figsize=render.figsize, dpi=render.dpi)
    _apply_dark(ax, render.dark)

    if E.shape[0] > 0:
        i = E[:,0]; j = E[:,1]
        for a, b in zip(i, j):
            ax.plot([P[a,0], P[b,0]], [P[a,1], P[b,1]],
                    color="white", alpha=0.08, lw=0.8)

    if E.shape[0] > 0 and sel.shape[0] == E.shape[0]:
        i = E[:,0]; j = E[:,1]
        dx = P[j,0] - P[i,0]
        dy = P[j,1] - P[i,1]
        ang = np.mod(np.arctan2(dy, dx), math.pi)
        c = plt.cm.hsv(ang / math.pi)
        idx = np.where(sel)[0]
        for t in idx:
            a = int(i[t]); b = int(j[t])
            ax.plot([P[a,0], P[b,0]], [P[a,1], P[b,1]],
                    color=c[t], alpha=float(render.sel_edge_alpha), lw=float(render.sel_edge_lw))

    if P.shape[0] > 0:
        ax.scatter(P[:,0], P[:,1], s=float(render.point_size), c="#66ccff", alpha=0.50, linewidths=0)

    extent = _extent_from_points(P)
    _draw_axes_grid(ax, extent, render)
    ax.set_title(title, color=("white" if render.dark else "black"), fontsize=render.title_size, pad=12)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=("black" if render.dark else "white"))
    plt.close(fig)

def render_heatmap(H: np.ndarray, extent, title: str, out_png: str, render: RenderSpec, cmap: str = "inferno"):
    fig, ax = plt.subplots(figsize=render.figsize, dpi=render.dpi)
    _apply_dark(ax, render.dark)
    xmin,xmax,ymin,ymax = extent
    im = ax.imshow(H, interpolation="nearest", cmap=cmap, origin="lower",
                   extent=(xmin,xmax,ymin,ymax))
    _draw_axes_grid(ax, extent, render)
    ax.set_title(title, color=("white" if render.dark else "black"), fontsize=render.title_size, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=("white" if render.dark else "black"))
    cb.outline.set_edgecolor("white" if render.dark else "black")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=("black" if render.dark else "white"))
    plt.close(fig)

def render_node_field(P: np.ndarray, values: np.ndarray, title: str, out_png: str, render: RenderSpec, cmap: str = "magma"):
    fig, ax = plt.subplots(figsize=render.figsize, dpi=render.dpi)
    _apply_dark(ax, render.dark)
    if P.shape[0] > 0:
        sc = ax.scatter(P[:,0], P[:,1], s=max(6.0, float(render.point_size)*1.05),
                        c=values, cmap=cmap, alpha=0.95, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=("white" if render.dark else "black"))
        cb.outline.set_edgecolor("white" if render.dark else "black")
    extent = _extent_from_points(P)
    _draw_axes_grid(ax, extent, render)
    ax.set_title(title, color=("white" if render.dark else "black"), fontsize=render.title_size, pad=12)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=("black" if render.dark else "white"))
    plt.close(fig)

def render_hist(data: np.ndarray, bins: int, title: str, xlabel: str, out_png: str, dark: bool = True):
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
    ax.set_facecolor("black" if dark else "white")
    tc = "white" if dark else "black"
    ax.tick_params(colors=tc)
    for s in ax.spines.values():
        s.set_color(tc)
    ax.hist(data, bins=bins)
    ax.set_title(title, color=tc)
    ax.set_xlabel(xlabel, color=tc)
    ax.set_ylabel("count", color=tc)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=("black" if dark else "white"))
    plt.close(fig)


# ----------------------------
# Energy simulation + feasibility
# ----------------------------

def simulate_shared_bus(selected_edges: int, mean_w: float, es: EnergySpec) -> Dict[str, float]:
    C = float(es.C_store_uF) * 1e-6
    V = float(es.V_init)
    Vlo = float(es.V_dump_lo)
    Vhi = float(es.V_dump_hi)
    eta = float(es.eta_dump)
    Ileak = float(es.I_leak_uA) * 1e-6
    dt = float(es.dt_s)
    T = float(es.sim_s)

    q_event = float(es.q_per_event_nC) * 1e-9
    f_edge = float(es.f_edge_hz)
    I_edge = f_edge * q_event * float(mean_w)
    Iin = float(selected_edges) * I_edge

    dumps = 0
    E_out = 0.0

    steps = int(max(1, round(T / dt)))
    for _ in range(steps):
        dV = (Iin - Ileak) * dt / max(C, 1e-12)
        V += dV

        if V >= Vhi:
            Ed = 0.5 * C * (Vhi*Vhi - Vlo*Vlo)
            if Ed > 0:
                E_out += eta * Ed
            V = Vlo
            dumps += 1

        if V < 0:
            V = 0.0

    dump_rate = dumps / max(T, 1e-12)
    avg_power = E_out / max(T, 1e-12)
    return {
        "avg_power_W": float(avg_power),
        "avg_power_uW": float(avg_power * 1e6),
        "dump_rate_hz": float(dump_rate),
        "dumps": int(dumps),
        "I_edge_nA": float(I_edge * 1e9),
        "Iin_uA": float(Iin * 1e6),
        "V_end": float(V),
    }

def energy_feasibility(selected_edges: int, mean_w: float, es: EnergySpec) -> Dict[str, float]:
    C = float(es.C_store_uF) * 1e-6
    V0 = float(es.V_init)
    Vhi = float(es.V_dump_hi)
    Ileak = float(es.I_leak_uA) * 1e-6
    T = float(es.sim_s)

    q_event = float(es.q_per_event_nC) * 1e-9
    f_edge = float(es.f_edge_hz)
    I_edge = f_edge * q_event * float(mean_w)

    # break-even N where Iin == Ileak
    N_break_even = Ileak / max(I_edge, 1e-18)

    # N to reach Vhi in time T (approx, ignoring dumps and using constant net I)
    dV = max(0.0, (Vhi - V0))
    I_needed = (C * dV / max(T, 1e-12)) + Ileak
    N_dump_in_T = I_needed / max(I_edge, 1e-18)

    return {
        "I_edge_nA": float(I_edge * 1e9),
        "N_break_even": float(N_break_even),
        "N_dump_in_T": float(N_dump_in_T),
        "selected_edges": float(selected_edges),
        "selected_over_break_even": float(selected_edges / max(N_break_even, 1e-12)),
        "selected_over_dump_need": float(selected_edges / max(N_dump_in_T, 1e-12)),
    }

def plot_line(x, y, xlabel, ylabel, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
    ax.plot(x, y, lw=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def run_energy_sweeps(
    base_edges_selected: int,
    base_mean_w: float,
    es: EnergySpec,
    sweep: SweepSpec,
    out_dir: str
) -> Dict[str, dict]:
    mkdirp(out_dir)
    results = {}

    C_list = list(sweep.C_store_uF_list)
    power_uW = []
    dump_hz = []
    for C in C_list:
        es2 = EnergySpec(**{**es.__dict__, "C_store_uF": float(C)})
        r = simulate_shared_bus(base_edges_selected, base_mean_w, es2)
        power_uW.append(r["avg_power_uW"])
        dump_hz.append(r["dump_rate_hz"])
    results["sweep_C_store"] = {
        "C_store_uF": C_list,
        "avg_power_uW": power_uW,
        "dump_rate_hz": dump_hz,
        "base": {"edges_selected": int(base_edges_selected), "mean_w": float(base_mean_w)}
    }
    plot_line(C_list, power_uW, "C_store_uF", "avg_power_uW", "Power vs C_store (shared bus)",
              os.path.join(out_dir, "power_vs_C_store.png"))
    plot_line(C_list, dump_hz, "C_store_uF", "dump_rate_hz", "Dump rate vs C_store (shared bus)",
              os.path.join(out_dir, "dump_rate_vs_C_store.png"))

    V_list = list(sweep.V_dump_hi_list)
    power_uW2 = []
    dump_hz2 = []
    for Vhi in V_list:
        es2 = EnergySpec(**{**es.__dict__, "V_dump_hi": float(Vhi)})
        r = simulate_shared_bus(base_edges_selected, base_mean_w, es2)
        power_uW2.append(r["avg_power_uW"])
        dump_hz2.append(r["dump_rate_hz"])
    results["sweep_V_dump_hi"] = {"V_dump_hi": V_list, "avg_power_uW": power_uW2, "dump_rate_hz": dump_hz2}
    plot_line(V_list, power_uW2, "V_dump_hi", "avg_power_uW", "Power vs V_dump_hi",
              os.path.join(out_dir, "power_vs_V_dump_hi.png"))
    plot_line(V_list, dump_hz2, "V_dump_hi", "dump_rate_hz", "Dump rate vs V_dump_hi",
              os.path.join(out_dir, "dump_rate_vs_V_dump_hi.png"))

    I_list = list(sweep.I_leak_uA_list)
    power_uW3 = []
    for Ileak in I_list:
        es2 = EnergySpec(**{**es.__dict__, "I_leak_uA": float(Ileak)})
        r = simulate_shared_bus(base_edges_selected, base_mean_w, es2)
        power_uW3.append(r["avg_power_uW"])
    results["sweep_I_leak"] = {"I_leak_uA": I_list, "avg_power_uW": power_uW3}
    plot_line(I_list, power_uW3, "I_leak_uA", "avg_power_uW", "Power vs leakage current",
              os.path.join(out_dir, "power_vs_I_leak.png"))

    E_list = list(sweep.edges_list)
    power_uW4 = []
    for ee in E_list:
        r = simulate_shared_bus(int(ee), base_mean_w, es)
        power_uW4.append(r["avg_power_uW"])
    results["sweep_edges"] = {"edges": E_list, "avg_power_uW": power_uW4}
    plot_line(E_list, power_uW4, "edges", "avg_power_uW", "Power vs selected edges (size proxy)",
              os.path.join(out_dir, "power_vs_edges.png"))

    return results


# ----------------------------
# Prototype device schematic
# ----------------------------

def render_device_schematic(out_dir: str, dark: bool = True):
    mkdirp(out_dir)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.set_facecolor("black" if dark else "white")
    ax.set_axis_off()

    fg = "white" if dark else "black"
    accent = "#66ccff" if dark else "#0055aa"
    warn = "#ffcc33"

    ax.add_patch(plt.Rectangle((0.06, 0.12), 0.58, 0.76, fill=False, ec=fg, lw=2))
    ax.text(0.07, 0.90, "Quasi-Cell Tile Sheet (AB / PEN / HYBRID)", color=fg, fontsize=13, fontweight="bold")

    for k in range(8):
        x0 = 0.09 + k * 0.065
        ax.add_patch(plt.Rectangle((x0, 0.16), 0.035, 0.68, fill=False, ec=accent, lw=1.2, alpha=0.85))
    ax.text(0.07, 0.08, "Micro-electrode bus strips (rectified cell taps)", color=fg, fontsize=10)

    ax.plot([0.64, 0.78], [0.50, 0.50], color=fg, lw=2)
    ax.text(0.66, 0.53, "Shared DC bus", color=fg, fontsize=10)

    ax.add_patch(plt.Rectangle((0.78, 0.58), 0.18, 0.18, fill=False, ec=fg, lw=2))
    ax.text(0.79, 0.78, "C_store", color=fg, fontsize=12, fontweight="bold")
    ax.text(0.79, 0.60, "μF–mF scale\nlow ESR", color=fg, fontsize=10)

    ax.add_patch(plt.Rectangle((0.78, 0.30), 0.18, 0.18, fill=False, ec=warn, lw=2))
    ax.text(0.79, 0.50, "Dump\nswitch", color=warn, fontsize=12, fontweight="bold")
    ax.text(0.79, 0.33, "V_hi/V_lo\ncomparator", color=fg, fontsize=10)

    ax.add_patch(plt.Rectangle((0.78, 0.08), 0.18, 0.16, fill=False, ec=accent, lw=2))
    ax.text(0.79, 0.25, "Load", color=accent, fontsize=12, fontweight="bold")
    ax.text(0.79, 0.10, "MCU / radio\nburst mode", color=fg, fontsize=10)

    ax.plot([0.87, 0.87], [0.58, 0.48], color=fg, lw=2)
    ax.plot([0.87, 0.87], [0.30, 0.24], color=fg, lw=2)
    ax.plot([0.87, 0.64], [0.48, 0.50], color=fg, lw=2)
    ax.plot([0.87, 0.64], [0.30, 0.50], color=warn, lw=1.6, alpha=0.9)

    ax.text(0.06, 0.01, "Prototype schematic: quasi-lattice harvest → shared bus → store → threshold dump bursts", color=fg, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prototype_device_schematic.png"), facecolor=("black" if dark else "white"))
    plt.close(fig)


# ----------------------------
# Domain mapping (Photonics / Phonons / Resistor)
# ----------------------------

def build_weighted_adjacency(n: int, E: np.ndarray, sel: np.ndarray, base_w: float, sel_boost: float):
    if E.shape[0] == 0 or n == 0:
        if _SCIPY_OK:
            return sp.csr_matrix((n, n), dtype=float)
        return np.zeros((n, n), dtype=float)

    i = E[:,0].astype(int)
    j = E[:,1].astype(int)
    w = np.full(E.shape[0], float(base_w), dtype=float)
    if sel is not None and sel.shape[0] == E.shape[0]:
        w = w + float(sel_boost) * sel.astype(float)

    ii = np.concatenate([i, j])
    jj = np.concatenate([j, i])
    ww = np.concatenate([w, w])

    if _SCIPY_OK:
        W = sp.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
        return W
    else:
        W = np.zeros((n, n), dtype=float)
        W[ii, jj] += ww
        return W

def weighted_laplacian(W):
    if _SCIPY_OK and sp.issparse(W):
        d = np.array(W.sum(axis=1)).reshape(-1)
        D = sp.diags(d, offsets=0, shape=W.shape, dtype=float)
        return D - W
    else:
        d = np.sum(W, axis=1)
        return np.diag(d) - W

def boundary_nodes_by_x(P: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    x = P[:,0]
    lo = np.quantile(x, frac)
    hi = np.quantile(x, 1.0 - frac)
    left = np.where(x <= lo)[0]
    right = np.where(x >= hi)[0]
    return left.astype(int), right.astype(int)

def safe_eigs_symmetric(M, k: int, which: str, max_dense_n: int):
    n = M.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float), None, {"method": "empty", "n": 0, "exact": True}

    # Dense path if feasible
    if (not _SCIPY_OK or not sp.issparse(M)) and n <= max_dense_n:
        vals, vecs = np.linalg.eigh(M)
        return vals, vecs, {"method": "dense_eigh", "n": n, "exact": True}

    # Sparse path
    if _SCIPY_OK and sp.issparse(M):
        if n <= 2:
            A = M.toarray()
            vals, vecs = np.linalg.eigh(A)
            return vals, vecs, {"method": "tiny_dense", "n": n, "exact": True}

        kk = int(min(max(2, k), n-1))
        try:
            vals, vecs = spla.eigsh(M, k=kk, which=which)
            vals = np.array(vals, dtype=float)
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
            return vals, vecs, {"method": "sparse_eigsh", "n": n, "k": int(kk), "which": which, "exact": False}
        except Exception as e:
            if n <= max_dense_n:
                A = M.toarray()
                vals, vecs = np.linalg.eigh(A)
                return vals, vecs, {"method": "fallback_dense_eigh", "n": n, "exact": True, "note": str(e)}
            return np.zeros((0,), dtype=float), None, {"method": "failed_sparse", "n": n, "exact": False, "error": str(e)}

    # No SciPy sparse, too big: approximate submatrix
    if n > max_dense_n:
        m = max_dense_n
        idx = np.linspace(0, n-1, m).astype(int)
        A = M[np.ix_(idx, idx)] if isinstance(M, np.ndarray) else np.array(M)[np.ix_(idx, idx)]
        vals, vecs = np.linalg.eigh(A)
        return vals, vecs, {"method": "approx_submatrix_dense", "n": n, "m": int(m), "exact": False}

    vals, vecs = np.linalg.eigh(M if isinstance(M, np.ndarray) else np.array(M))
    return vals, vecs, {"method": "dense_eigh", "n": n, "exact": True}

def photonics_domain(P: np.ndarray, E: np.ndarray, sel: np.ndarray, out_dir: str, render: RenderSpec, spec: PhotonicsSpec) -> Dict[str, object]:
    mkdirp(out_dir)
    n = P.shape[0]
    W = build_weighted_adjacency(n, E, sel, base_w=spec.base_coupling, sel_boost=spec.sel_boost)

    # Eigenvalues of adjacency -> proxy spectrum
    if _SCIPY_OK and sp.issparse(W):
        k = min(int(spec.k_sparse), max(2, n-1))
        vals, vecs, meta = safe_eigs_symmetric(W, k=k, which="LA", max_dense_n=int(spec.max_dense_n))
        approx = True
    else:
        vals, vecs, meta = safe_eigs_symmetric(W, k=min(int(spec.k_sparse), max(2,n-1)), which="LA", max_dense_n=int(spec.max_dense_n))
        approx = (meta.get("exact", False) is False)

    freqs = spec.w0 + spec.kappa * vals

    # Boundary loss proxy
    x = P[:,0]; y = P[:,1]
    r = np.sqrt(x*x + y*y)
    r_thr = np.quantile(r, 0.90) if n > 10 else np.max(r) if n else 0.0
    boundary = (r >= r_thr).astype(float)
    gamma_i = spec.boundary_loss_gamma * boundary

    mode_outputs = []
    if vecs is not None and vecs.shape[0] == n and vecs.shape[1] >= 1:
        mcount = min(int(spec.modes_to_render), vecs.shape[1])
        for mi in range(mcount):
            v = vecs[:, -(mi+1)]
            amp = np.abs(v)
            amp = amp / (float(np.max(amp)) + 1e-12)
            loss = float(np.sum((np.abs(v)**2) * gamma_i))
            q_proxy = float(1.0 / max(loss, 1e-12))
            png = f"photon_mode_{mi}.png"
            render_node_field(P, amp, f"Photon mode {mi} | Q_proxy≈{q_proxy:.2e}", os.path.join(out_dir, png), render, cmap="magma")
            mode_outputs.append({"mode": mi, "png": png, "q_proxy": q_proxy, "loss": loss})

    spec_png = "photon_spectrum.png"
    if freqs.size > 0:
        render_hist(freqs, bins=int(spec.spectrum_bins), title=f"Photonics spectrum ({'approx' if approx else 'exact'})",
                    xlabel="frequency (arb THz)", out_png=os.path.join(out_dir, spec_png), dark=render.dark)

    out = {
        "domain": "photonics",
        "n": int(n),
        "m_edges": int(E.shape[0]),
        "method": meta,
        "approx_spectrum": bool(approx),
        "freqs_summary": {"min": float(np.min(freqs)) if freqs.size else float("nan"),
                          "max": float(np.max(freqs)) if freqs.size else float("nan"),
                          **robust_percentiles(freqs, (10,50,90))},
        "files": {"spectrum_png": spec_png, "modes": mode_outputs},
        "spec": spec.__dict__,
    }
    save_json(os.path.join(out_dir, "photonics.json"), out)
    return out

def phonon_domain(P: np.ndarray, E: np.ndarray, sel: np.ndarray, out_dir: str, render: RenderSpec, spec: PhononSpec) -> Dict[str, object]:
    mkdirp(out_dir)
    n = P.shape[0]
    W = build_weighted_adjacency(n, E, sel, base_w=spec.base_stiffness, sel_boost=spec.sel_boost)
    L = weighted_laplacian(W)

    if _SCIPY_OK and sp.issparse(L):
        k = min(int(spec.k_sparse), max(2, n-1))
        vals, vecs, meta = safe_eigs_symmetric(L, k=k, which="SM", max_dense_n=int(spec.max_dense_n))
        approx = True
    else:
        vals, vecs, meta = safe_eigs_symmetric(L, k=min(int(spec.k_sparse), max(2,n-1)), which="SA", max_dense_n=int(spec.max_dense_n))
        approx = (meta.get("exact", False) is False)

    vals = np.maximum(vals, 0.0)
    omega = np.sqrt((spec.k_spring / max(spec.mass, 1e-12)) * vals)

    mode_outputs = []
    if vecs is not None and vecs.shape[0] == n and vecs.shape[1] >= 2:
        # skip first near-zero mode (one component)
        mcount = min(int(spec.modes_to_render), max(0, vecs.shape[1]-1))
        for mi in range(mcount):
            v = vecs[:, mi+1]
            amp = np.abs(v)
            amp = amp / (float(np.max(amp)) + 1e-12)
            png = f"phonon_mode_{mi}.png"
            render_node_field(P, amp, f"Phonon mode {mi} | ω≈{float(omega[mi+1]):.3f}", os.path.join(out_dir, png), render, cmap="viridis")
            mode_outputs.append({"mode": mi, "png": png, "omega": float(omega[mi+1])})

    spec_png = "phonon_spectrum.png"
    if omega.size > 0:
        render_hist(omega, bins=int(spec.spectrum_bins), title=f"Phonon spectrum ({'approx' if approx else 'exact'})",
                    xlabel="omega (arb)", out_png=os.path.join(out_dir, spec_png), dark=render.dark)

    out = {
        "domain": "phonons",
        "n": int(n),
        "m_edges": int(E.shape[0]),
        "method": meta,
        "approx_spectrum": bool(approx),
        "omega_summary": {"min": float(np.min(omega)) if omega.size else float("nan"),
                          "max": float(np.max(omega)) if omega.size else float("nan"),
                          **robust_percentiles(omega, (10,50,90))},
        "files": {"spectrum_png": spec_png, "modes": mode_outputs},
        "spec": spec.__dict__,
    }
    save_json(os.path.join(out_dir, "phonons.json"), out)
    return out

def resistor_domain(P: np.ndarray, E: np.ndarray, sel: np.ndarray, out_dir: str, render: RenderSpec, spec: ResistorSpec) -> Dict[str, object]:
    mkdirp(out_dir)
    n_full = P.shape[0]
    if n_full == 0 or E.shape[0] == 0:
        out = {"domain": "resistor", "n": int(n_full), "m_edges": int(E.shape[0]), "ok": False, "reason": "empty"}
        save_json(os.path.join(out_dir, "resistor.json"), out)
        return out

    left, right = boundary_nodes_by_x(P, frac=float(spec.electrode_frac))

    # Optionally restrict to electrode-connected subgraph to avoid singular L_ff
    keep_mask = np.ones(n_full, dtype=bool)
    if spec.use_electrode_component_only:
        keep_mask = electrode_connected_mask(P, E, left, right)
        if not np.any(keep_mask):
            out = {"domain": "resistor", "n": int(n_full), "m_edges": int(E.shape[0]), "ok": False, "reason": "no electrode-connected nodes"}
            save_json(os.path.join(out_dir, "resistor.json"), out)
            return out

    # Build induced subgraph
    idx_map = -np.ones(n_full, dtype=int)
    keep_idx = np.where(keep_mask)[0].astype(int)
    idx_map[keep_idx] = np.arange(keep_idx.size, dtype=int)

    Pk = P[keep_idx]
    # remap edges
    Em = []
    selm = []
    for t, (a, b) in enumerate(E):
        a = int(a); b = int(b)
        aa = idx_map[a]; bb = idx_map[b]
        if aa >= 0 and bb >= 0 and aa != bb:
            Em.append((aa, bb))
            selm.append(bool(sel[t]) if sel is not None and sel.shape[0] == E.shape[0] else False)
    Ek = np.array(Em, dtype=np.int32) if Em else np.zeros((0,2), dtype=np.int32)
    selk = np.array(selm, dtype=bool) if selm else np.zeros((Ek.shape[0],), dtype=bool)

    # electrodes in subgraph
    leftk = idx_map[left]
    rightk = idx_map[right]
    leftk = leftk[leftk >= 0]
    rightk = rightk[rightk >= 0]
    if leftk.size == 0 or rightk.size == 0:
        out = {
            "domain": "resistor",
            "ok": False,
            "reason": "electrodes missing after pruning",
            "n_full": int(n_full),
            "n_used": int(Pk.shape[0]),
            "m_edges_used": int(Ek.shape[0]),
            "pruned_frac": float(1.0 - (Pk.shape[0] / max(n_full,1))),
        }
        save_json(os.path.join(out_dir, "resistor.json"), out)
        return out

    # Build conductance adjacency on used subgraph
    W = build_weighted_adjacency(Pk.shape[0], Ek, selk, base_w=spec.base_g, sel_boost=spec.sel_boost)
    L = weighted_laplacian(W)

    fixed = np.zeros(Pk.shape[0], dtype=bool)
    fixed[leftk] = True
    fixed[rightk] = True
    V = np.zeros(Pk.shape[0], dtype=float)
    V[leftk] = float(spec.V_left)
    V[rightk] = float(spec.V_right)

    free = np.where(~fixed)[0]
    if free.size == 0:
        out = {"domain": "resistor", "ok": False, "reason": "no free nodes", "n_used": int(Pk.shape[0])}
        save_json(os.path.join(out_dir, "resistor.json"), out)
        return out

    ok = False
    method = "none"
    Vsol = V.copy()

    # Solve: L_ff V_f = - L_fb V_b
    try:
        if _SCIPY_OK and sp.issparse(L):
            Lff = L[free][:, free].tocsr()
            Lfb = L[free][:, fixed].tocsr()
            Vb = V[fixed]
            rhs = - (Lfb @ Vb)
            if float(spec.shunt_eps) > 0.0:
                Lff = Lff + float(spec.shunt_eps) * sp.eye(Lff.shape[0], format="csr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Vf = spla.spsolve(Lff, rhs)
            Vsol[free] = np.array(Vf, dtype=float)
            ok = True
            method = "spsolve" + (f"+shunt{spec.shunt_eps:g}" if float(spec.shunt_eps) > 0 else "")
        else:
            Ld = np.array(L, dtype=float)
            Lff = Ld[np.ix_(free, free)]
            Lfb = Ld[np.ix_(free, np.where(fixed)[0])]
            rhs = - Lfb @ V[fixed]
            if float(spec.shunt_eps) > 0.0:
                Lff = Lff + float(spec.shunt_eps) * np.eye(Lff.shape[0])
            Vf = np.linalg.solve(Lff, rhs)
            Vsol[free] = Vf
            ok = True
            method = "dense_solve" + (f"+shunt{spec.shunt_eps:g}" if float(spec.shunt_eps) > 0 else "")
    except Exception as e:
        ok = False
        method = f"solve_failed:{e}"

    Geff = float("nan")
    I_total = float("nan")
    pot_png = None
    if ok:
        # I_left = sum_{i in left} sum_j g_ij (V_i - V_j)
        I = 0.0
        if _SCIPY_OK and sp.issparse(W):
            Wcsr = W.tocsr()
            for i in leftk:
                start = Wcsr.indptr[i]
                end = Wcsr.indptr[i+1]
                js = Wcsr.indices[start:end]
                ws = Wcsr.data[start:end]
                I += float(np.sum(ws * (Vsol[i] - Vsol[js])))
        else:
            A = W
            for i in leftk:
                js = np.where(A[i] != 0)[0]
                ws = A[i, js]
                I += float(np.sum(ws * (Vsol[i] - Vsol[js])))

        I_total = float(I)
        dV = float(spec.V_left - spec.V_right)
        Geff = float(I_total / max(dV, 1e-12))

        pot_png = "resistor_potential.png"
        render_node_field(Pk, Vsol, f"Resistor potential | G_eff≈{Geff:.6f}", os.path.join(out_dir, pot_png), render, cmap="plasma")

    out = {
        "domain": "resistor",
        "ok": bool(ok),
        "method": method,
        "n_full": int(n_full),
        "n_used": int(Pk.shape[0]),
        "m_edges_full": int(E.shape[0]),
        "m_edges_used": int(Ek.shape[0]),
        "pruned_frac": float(1.0 - (Pk.shape[0] / max(n_full,1))),
        "electrodes": {"left_n": int(leftk.size), "right_n": int(rightk.size),
                       "electrode_frac": float(spec.electrode_frac),
                       "V_left": float(spec.V_left), "V_right": float(spec.V_right)},
        "I_total": float(I_total),
        "G_eff": float(Geff),
        "files": {"potential_png": pot_png} if pot_png else {},
        "spec": spec.__dict__,
    }
    save_json(os.path.join(out_dir, "resistor.json"), out)
    return out

def run_domain_suite(job_name: str, P: np.ndarray, E: np.ndarray, sel: np.ndarray,
                     out_dir: str, render: RenderSpec, suite: DomainSuiteSpec) -> Dict[str, object]:
    mkdirp(out_dir)
    out = {"job": job_name, "domains": {}, "scipy": _SCIPY_OK}

    if suite.parallel:
        try:
            import concurrent.futures as cf
            futures = {}
            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                if suite.run_photonics:
                    futures["photonics"] = ex.submit(photonics_domain, P, E, sel, os.path.join(out_dir, "photonics"), render, suite.phot)
                if suite.run_phonons:
                    futures["phonons"] = ex.submit(phonon_domain, P, E, sel, os.path.join(out_dir, "phonons"), render, suite.phon)
                if suite.run_resistor:
                    futures["resistor"] = ex.submit(resistor_domain, P, E, sel, os.path.join(out_dir, "resistor"), render, suite.res)
                for k, fut in futures.items():
                    out["domains"][k] = fut.result()
            save_json(os.path.join(out_dir, "domain_suite.json"), out)
            return out
        except Exception as e:
            out["parallel_failed"] = str(e)

    if suite.run_photonics:
        out["domains"]["photonics"] = photonics_domain(P, E, sel, os.path.join(out_dir, "photonics"), render, suite.phot)
    if suite.run_phonons:
        out["domains"]["phonons"] = phonon_domain(P, E, sel, os.path.join(out_dir, "phonons"), render, suite.phon)
    if suite.run_resistor:
        out["domains"]["resistor"] = resistor_domain(P, E, sel, os.path.join(out_dir, "resistor"), render, suite.res)

    save_json(os.path.join(out_dir, "domain_suite.json"), out)
    return out


# ----------------------------
# Atlas runner + report
# ----------------------------

def run_geom_atlas(
    jobs: List[GeometryJob],
    out_dir: str,
    domains_out_dir: str,
    energy_out_dir: str,
    device_out_dir: str,
    seed: int = 0,
    suite: Optional[DomainSuiteSpec] = None,
    es: Optional[EnergySpec] = None,
) -> Dict[str, object]:
    mkdirp(out_dir)
    mkdirp(domains_out_dir)
    mkdirp(energy_out_dir)
    mkdirp(device_out_dir)

    manifest = {
        "version": "qc_geom_energy_v0.28.0",
        "created_ms": _now_ms(),
        "seed": int(seed),
        "dirs": {
            "atlas": out_dir,
            "domains": domains_out_dir,
            "energy": energy_out_dir,
            "device": device_out_dir,
        },
        "scipy_available": _SCIPY_OK,
        "jobs": {},
        "files": []
    }

    report_jobs = {}

    cache_geom: Dict[str, Tuple[np.ndarray, np.ndarray, dict]] = {}

    for job in jobs:
        t0 = time.time()

        # ---- build points
        if job.kind == "AB":
            assert job.ab is not None
            P, diag = make_ab_patch(job.ab, bbox_R=float(job.placement.bbox_R), seed=seed)
            # AB edges: two shells ~ 1 and sqrt(2)
            E, edge_meta = build_edges_auto(
                P, "shell",
                shell_lengths=[1.0, math.sqrt(2.0)],
                edge_tol=float(job.ab.edge_tol),
                knn_k=6,
                knn_rmax=1.9,
                max_deg=int(job.ab.max_deg),
                seed=seed+13,
            )
            diag["edges"] = edge_meta

        elif job.kind == "PENW":
            assert job.penw is not None
            P, stats, pen_diag = make_pen_wave_points(job.penw, seed_override=seed + int(job.penw.seed))
            # robust edges for pen: kNN by default
            E, edge_meta = build_edges_auto(
                P, job.penw.edge_policy,
                shell_lengths=[1.0, 1.618],
                edge_tol=float(job.penw.edge_tol),
                knn_k=int(job.penw.knn_k),
                knn_rmax=float(job.penw.knn_rmax),
                max_deg=int(job.penw.max_deg),
                seed=seed+17,
            )
            diag = {
                "nn": {
                    "median": float(estimate_nn_median(P, seed=seed)),
                    **robust_percentiles(_nearest_neighbor_dist(P), (10,50,90))
                },
                "pen": stats,
                "pen_diag": pen_diag,
                "edges": edge_meta
            }

        elif job.kind == "PENCP":
            assert job.pencp is not None
            P, diag = make_pen_cp_points(job.pencp)
            E, edge_meta = build_edges_auto(
                P, job.pencp.edge_policy,
                shell_lengths=[1.0, 1.618],
                edge_tol=float(job.pencp.edge_tol),
                knn_k=int(job.pencp.knn_k),
                knn_rmax=float(job.pencp.knn_rmax),
                max_deg=int(job.pencp.max_deg),
                seed=seed+19,
            )
            diag["edges"] = edge_meta

        elif job.kind == "SQUARE":
            assert job.square is not None
            P, E, diag = make_square(job.square)

        elif job.kind == "HYBRID":
            assert job.hybrid is not None
            base = cache_geom.get(job.hybrid.base_name)
            if base is None:
                raise RuntimeError(f"Hybrid base '{job.hybrid.base_name}' not found. Ensure AB job runs before HYBRID.")
            P, E, base_diag = base
            diag = {
                "hybrid_base": job.hybrid.base_name,
                "base_diag": base_diag,
                "field_map_scale": float(job.hybrid.field_map_scale),
                "pen_field": {
                    "L": float(job.hybrid.pen_field_L),
                    "dx": float(job.hybrid.pen_field_dx),
                    "k": float(job.hybrid.pen_field_k),
                    "seed": int(job.hybrid.pen_field_seed),
                }
            }
        else:
            raise ValueError(f"Unknown job.kind: {job.kind}")

        # cache AB for HYBRID
        if job.kind == "AB":
            cache_geom[job.name] = (P, E, diag)

        # ---- HYBRID field values at edge midpoints
        mids = edge_midpoints(P, E)
        field_vals = None
        if job.kind == "HYBRID":
            xs, ys, F = pen_wave_field_grid(
                L=float(job.hybrid.pen_field_L),
                dx=float(job.hybrid.pen_field_dx),
                k=float(job.hybrid.pen_field_k),
                seed=int(job.hybrid.pen_field_seed),
            )
            Q = mids * float(job.hybrid.field_map_scale)
            field_vals = sample_field_at_points(xs, ys, F, Q)

        # ---- deploy selection
        sel_mask, dep_diag = deploy_edges_mask(P, E, job.kind, job.deploy, field_values_at_mid=field_vals, seed=seed+7)

        # ---- connectivity stats
        gstats = component_stats(P.shape[0], E)
        # electrode connectivity (for resistor feasibility)
        left, right = boundary_nodes_by_x(P, frac=0.06) if P.shape[0] > 0 else (np.zeros((0,),dtype=int), np.zeros((0,),dtype=int))
        econn = electrode_connected_mask(P, E, left, right) if P.shape[0] > 0 else np.zeros((0,), dtype=bool)
        econn_frac = float(np.mean(econn)) if econn.size else 0.0

        # ---- heatmaps
        Hw, Hc, extent = edge_heatmaps(P, E, sel_mask, nbins=80)

        # ---- renders
        patch_png = f"{job.name}_patch.png"
        deploy_png = f"{job.name}_deploy.png"
        heat_motif_png = f"{job.name}_heat_motif.png"
        heat_edges_png = f"{job.name}_heat_edges.png"

        render_patch(P, E, f"{job.name} — patch (all edges)", os.path.join(out_dir, patch_png), job.render)
        render_deploy(P, E, sel_mask, f"{job.name} — selected edges (deployment mask)", os.path.join(out_dir, deploy_png), job.render)
        render_heatmap(Hw, extent, f"{job.name} — heatmap: Σ motif weight", os.path.join(out_dir, heat_motif_png), job.render)
        render_heatmap(Hc, extent, f"{job.name} — heatmap: edge count", os.path.join(out_dir, heat_edges_png), job.render)

        # ---- domains
        domain_summary = None
        if suite is not None:
            dom_dir = os.path.join(domains_out_dir, job.name)
            domain_summary = run_domain_suite(job.name, P, E, sel_mask, dom_dir, job.render, suite)

        # ---- energy per job (hard numbers)
        energy_job = None
        if es is not None:
            mean_w = float(dep_diag.get("motif_selected_mean", 0.80))
            if not np.isfinite(mean_w):
                mean_w = 0.80
            sim = simulate_shared_bus(int(dep_diag["selected"]), mean_w, es)
            feas = energy_feasibility(int(dep_diag["selected"]), mean_w, es)
            energy_job = {"mean_w": float(mean_w), "sim": sim, "feasibility": feas}

        runtime_s = float(time.time() - t0)

        deg_mean = float((2.0*E.shape[0]) / max(P.shape[0], 1))
        span = float(2.0 * math.sqrt(np.max(np.sum(P**2, axis=1))) if P.shape[0] else 0.0)

        manifest["jobs"][job.name] = {
            "kind": job.kind,
            "points": int(P.shape[0]),
            "edges": int(E.shape[0]),
            "deg_mean": float(deg_mean),
            "span": float(span),
            "components": gstats,
            "electrode_connected_frac": float(econn_frac),
            "deploy": dep_diag,
            "diag": diag,
            "runtime_s": runtime_s,
            "files": {
                "patch_png": patch_png,
                "deploy_png": deploy_png,
                "heat_motif_png": heat_motif_png,
                "heat_edges_png": heat_edges_png,
            },
            "domains": domain_summary["domains"] if domain_summary else None,
            "energy": energy_job,
        }

        report_jobs[job.name] = {
            "kind": job.kind,
            "points": int(P.shape[0]),
            "edges": int(E.shape[0]),
            "deg_mean": float(deg_mean),
            "components": gstats,
            "electrode_connected_frac": float(econn_frac),
            "deploy": dep_diag,
            "energy": energy_job,
            "domains": domain_summary["domains"] if domain_summary else None,
        }

        manifest["files"].extend([patch_png, deploy_png, heat_motif_png, heat_edges_png])

    # Device schematic
    render_device_schematic(device_out_dir, dark=True)
    manifest["files"].append(os.path.join(device_out_dir, "prototype_device_schematic.png"))

    # Write manifest + report
    manifest_path = os.path.join(out_dir, "qc_geom_energy_manifest.json")
    save_json(manifest_path, manifest)

    report = {
        "version": "qc_report_v0.28.0",
        "created_ms": _now_ms(),
        "seed": int(seed),
        "scipy_available": _SCIPY_OK,
        "jobs": report_jobs,
    }
    report_json = os.path.join(out_dir, "qc_report.json")
    save_json(report_json, report)

    # Short MD summary
    lines = []
    lines.append("# QC Geometric Energy Cell — Report v0.28.0\n")
    lines.append(f"- created_ms: {report['created_ms']}\n")
    lines.append(f"- scipy_available: {report['scipy_available']}\n")
    lines.append("\n## Jobs\n")
    for name, j in report_jobs.items():
        lines.append(f"### {name}\n")
        lines.append(f"- kind: {j['kind']}\n")
        lines.append(f"- points: {j['points']}  edges: {j['edges']}  deg_mean: {j['deg_mean']:.2f}\n")
        c = j["components"]
        lines.append(f"- components: {c['components']}  largest_frac: {c['largest_frac']:.3f}  top5: {c['size_top5']}\n")
        lines.append(f"- electrode_connected_frac: {j['electrode_connected_frac']:.3f}\n")
        d = j["deploy"]
        lines.append(f"- selected_edges: {d['selected']}  deploy_frac: {d['deploy_frac']:.3f}\n")
        lines.append(f"- motif_selected_mean: {d.get('motif_selected_mean', float('nan')):.3f}\n")
        if j["energy"] is not None:
            sim = j["energy"]["sim"]
            feas = j["energy"]["feasibility"]
            lines.append(f"- energy(sim): dumps={sim['dumps']}  dump_rate_hz={sim['dump_rate_hz']:.4f}  avg_power_uW={sim['avg_power_uW']:.4f}\n")
            lines.append(f"- energy(feas): N_break_even≈{feas['N_break_even']:.1f}  N_dump_in_T≈{feas['N_dump_in_T']:.1f}  selected/N_dump≈{feas['selected_over_dump_need']:.4f}\n")
        if j["domains"] is not None:
            if "resistor" in j["domains"]:
                r = j["domains"]["resistor"]
                if r.get("ok", False):
                    lines.append(f"- resistor: G_eff≈{r.get('G_eff', float('nan')):.6f} (method={r.get('method')})\n")
                else:
                    lines.append(f"- resistor: FAILED (method={r.get('method')}, reason={r.get('reason','')})\n")
        lines.append("\n")
    report_md = os.path.join(out_dir, "qc_report.md")
    save_text(report_md, "".join(lines))

    return {
        "manifest_path": manifest_path,
        "report_json": report_json,
        "report_md": report_md,
        "manifest": manifest,
        "report": report,
    }


# ----------------------------
# Bundling / ZIP + Colab download
# ----------------------------

def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False

def colab_download(path: str):
    try:
        from google.colab import files
        files.download(path)
    except Exception as e:
        print(f"[warn] colab download failed: {e}")

def zip_dir(z: zipfile.ZipFile, folder: str, arc_prefix: str):
    for root, _, files in os.walk(folder):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, folder)
            z.write(p, arcname=os.path.join(arc_prefix, rel))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_geom_atlas", help="output dir for geometry atlas renders + report")
    ap.add_argument("--domains_out", default="out_domains", help="output dir for domain suite outputs")
    ap.add_argument("--energy_out", default="out_energy_sweeps", help="output dir for energy sweeps")
    ap.add_argument("--device_out", default="out_device", help="output dir for prototype device images")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zip", action="store_true", help="zip outputs into one bundle")
    ap.add_argument("--zip_name", default="", help="optional zip filename override")
    ap.add_argument("--auto_download", action="store_true", help="auto-download zip in Colab")
    ap.add_argument("--autorun", action="store_true", help="full automated run: implies --zip --auto_download")
    ap.add_argument("--quick", action="store_true", help="smaller/faster run")
    ap.add_argument("--no_parallel", action="store_true", help="disable parallel domain suite")
    ap.add_argument("--res_shunt", type=float, default=0.0, help="resistor shunt eps regularization (0 disables)")
    args = ap.parse_args(_strip_ipykernel_argv(list(os.sys.argv[1:])))

    if args.autorun:
        args.zip = True
        args.auto_download = True

    mkdirp(args.out)
    mkdirp(args.domains_out)
    mkdirp(args.energy_out)
    mkdirp(args.device_out)

    # Visual style (axes + comprehensive grids)
    r = RenderSpec(
        dark=True, figsize=(8.2, 8.2), dpi=170,
        show_axes=True, show_grid=True,
        grid_major=5.0, grid_minor=1.0,
        grid_alpha_major=0.45, grid_alpha_minor=0.16,
        point_size=7.0, point_alpha=0.75,
        edge_lw=1.0, edge_alpha=0.28,
        sel_edge_lw=3.0, sel_edge_alpha=0.95,
        title_size=18, equal_aspect=True,
    )

    d = DeploySpec(deploy_frac=0.08, nbins=12, phase=0.6, jitter=0.10, mode="topk", field_gain=0.65)

    suite = DomainSuiteSpec(
        run_photonics=True,
        run_phonons=True,
        run_resistor=True,
        parallel=(not args.no_parallel),
        phot=PhotonicsSpec(k_sparse=140 if args.quick else 220),
        phon=PhononSpec(k_sparse=140 if args.quick else 220),
        res=ResistorSpec(
            base_g=1.0, sel_boost=2.2, electrode_frac=0.06, V_left=1.0, V_right=0.0,
            use_electrode_component_only=True,
            shunt_eps=float(args.res_shunt),
        ),
    )

    # Energy spec (you will tune these once we have comparative hard numbers)
    es = EnergySpec(
        sim_s=30.0,
        dt_s=0.001,
        C_store_uF=330.0,
        V_init=3.0,
        V_dump_lo=3.2,
        V_dump_hi=5.2,
        eta_dump=0.85,
        I_leak_uA=2.0,
        f_edge_hz=0.9,
        q_per_event_nC=1.4,
    )

    # Jobs: AB, PENW (multiple), PENCP, Square control, Hybrid
    jobs: List[GeometryJob] = []

    # AB: bigger and denser
    ab4 = ABSpec(window_w=1.96, edge_tol=0.018, max_deg=6, R_lattice=(12 if args.quick else 14))
    ab5 = ABSpec(window_w=2.15, edge_tol=0.018, max_deg=6, R_lattice=(13 if args.quick else 15))

    jobs.append(GeometryJob(name="AB_n4_4", kind="AB", placement=PlacementSpec(bbox_R=22.0 if args.quick else 26.0), render=r, deploy=d, ab=ab4))
    jobs.append(GeometryJob(name="AB_n4_5", kind="AB", placement=PlacementSpec(bbox_R=22.0 if args.quick else 28.0), render=r, deploy=d, ab=ab5))

    # PEN wave-field (robust)
    pen_small = PenWaveSpec(
        L=18.0, dx=0.25, k=1.0, seed=9,
        peak_quantile=0.86, r_min=0.35,
        topK=(1500 if args.quick else 1900),
        min_peaks=(900 if args.quick else 1100),
        edge_policy="knn", knn_k=6, knn_rmax=1.8, max_deg=8, edge_tol=0.03,
    )
    pen_med = PenWaveSpec(
        L=18.0, dx=0.25, k=1.0, seed=9,
        peak_quantile=0.83, r_min=0.35,
        topK=(2100 if args.quick else 2600),
        min_peaks=(1150 if args.quick else 1500),
        edge_policy="knn", knn_k=6, knn_rmax=1.8, max_deg=8, edge_tol=0.03,
    )
    pen_large = PenWaveSpec(
        L=18.0, dx=0.25, k=1.0, seed=9,
        peak_quantile=0.80, r_min=0.35,
        topK=(2600 if args.quick else 3200),
        min_peaks=(1500 if args.quick else 2100),
        edge_policy="knn", knn_k=6, knn_rmax=1.8, max_deg=8, edge_tol=0.03,
    )

    jobs.append(GeometryJob(name="PEN_wave_small", kind="PENW", placement=PlacementSpec(bbox_R=30.0), render=r, deploy=d, penw=pen_small))
    jobs.append(GeometryJob(name="PEN_wave_med", kind="PENW", placement=PlacementSpec(bbox_R=30.0), render=r, deploy=d, penw=pen_med))
    jobs.append(GeometryJob(name="PEN_wave_large", kind="PENW", placement=PlacementSpec(bbox_R=30.0), render=r, deploy=d, penw=pen_large))

    # PEN CP variant (edges fixed by knn fallback)
    pencp = PenCPSpec(R_lattice=(5 if args.quick else 6), window_w=2.2, bbox_R=24.0,
                      max_deg=8, edge_policy="knn", knn_k=6, knn_rmax=1.8, edge_tol=0.03, seed=4)
    jobs.append(GeometryJob(name="PEN_cp_variant", kind="PENCP", placement=PlacementSpec(bbox_R=24.0), render=r, deploy=d, pencp=pencp))

    # Square control
    sq = SquareSpec(spacing=1.0, half_extent=(14 if args.quick else 18))
    jobs.append(GeometryJob(name="SQUARE_ctrl", kind="SQUARE", placement=PlacementSpec(bbox_R=18.0), render=r, deploy=DeploySpec(deploy_frac=0.06), square=sq))

    # Hybrid: AB base driven by pen field on edge midpoints
    hy = HybridSpec(base_name="AB_n4_4", field_map_scale=0.78, pen_field_L=18.0, pen_field_dx=0.25, pen_field_k=1.0, pen_field_seed=9)
    jobs.append(GeometryJob(
        name="HYBRID_AB4_PENmed",
        kind="HYBRID",
        placement=PlacementSpec(bbox_R=26.0),
        render=r,
        deploy=DeploySpec(deploy_frac=0.10, nbins=12, phase=0.6, jitter=0.10, mode="topk", field_gain=0.70),
        hybrid=hy
    ))

    t0 = time.time()
    atlas = run_geom_atlas(
        jobs=jobs,
        out_dir=args.out,
        domains_out_dir=args.domains_out,
        energy_out_dir=args.energy_out,
        device_out_dir=args.device_out,
        seed=args.seed,
        suite=suite,
        es=es,
    )

    # Energy sweeps base = HYBRID if present else first
    manifest = atlas["manifest"]
    base_name = "HYBRID_AB4_PENmed" if "HYBRID_AB4_PENmed" in manifest["jobs"] else list(manifest["jobs"].keys())[0]
    base_edges_selected = int(manifest["jobs"][base_name]["deploy"]["selected"])
    base_mean_w = float(manifest["jobs"][base_name]["deploy"].get("motif_selected_mean", 0.80))
    if not np.isfinite(base_mean_w):
        base_mean_w = 0.80

    sweeps = run_energy_sweeps(base_edges_selected, base_mean_w, es, SweepSpec(), out_dir=args.energy_out)
    sweeps_path = os.path.join(args.energy_out, "qc_energy_sweeps.json")
    save_json(sweeps_path, {
        "version": "qc_energy_sweeps_v0.28.0",
        "created_ms": _now_ms(),
        "base_job": base_name,
        "base_edges_selected": base_edges_selected,
        "base_mean_w": base_mean_w,
        "energy_spec": es.__dict__,
        "sweeps": sweeps
    })

    # Bundle descriptor
    bundle = {
        "version": "qc_geom_energy_domains_bundle_v0.28.0",
        "created_ms": _now_ms(),
        "runtime_s": float(time.time() - t0),
        "manifest_path": atlas["manifest_path"],
        "report_json": atlas["report_json"],
        "report_md": atlas["report_md"],
        "energy_sweep_json": sweeps_path,
        "dirs": {
            "atlas": args.out,
            "domains": args.domains_out,
            "energy": args.energy_out,
            "device": args.device_out,
        },
        "notes": {
            "domains": ["photonics (tight-binding)", "phonons (laplacian)", "resistor network (conductance)"],
            "resistor_fix": "electrode-component pruning + optional shunt eps",
            "penw_fix": "fallback_full_field + adaptive quantile + adaptive r_min",
            "edges_fix": "AB shells + knn fallback; PEN uses knn",
            "scipy_available": _SCIPY_OK
        }
    }
    bundle_path = os.path.join(".", "qc_geom_energy_domains_bundle.json")
    save_json(bundle_path, bundle)

    # ZIP
    zip_path = None
    if args.zip:
        if args.zip_name.strip():
            zip_path = args.zip_name.strip()
        else:
            zip_path = f"qc_geom_energy_domains_bundle_v0.28.0_{int(time.time())}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            zip_dir(z, args.out, "atlas")
            zip_dir(z, args.domains_out, "domains")
            zip_dir(z, args.energy_out, "energy")
            zip_dir(z, args.device_out, "device")
            z.write(bundle_path, arcname=os.path.basename(bundle_path))

    # Console summary (hard numbers)
    print(f"[ok] wrote manifest: {atlas['manifest_path']}")
    print(f"[ok] wrote report json: {atlas['report_json']}")
    print(f"[ok] wrote report md: {atlas['report_md']}")
    print(f"[ok] wrote energy sweeps: {sweeps_path}")
    print(f"[ok] wrote bundle: {bundle_path}")
    if zip_path:
        print(f"[ok] wrote zip: {zip_path}")
    print(f"[runtime_s] {bundle['runtime_s']:.3f}")
    for k, v in manifest["jobs"].items():
        c = v["components"]
        print(f" - {k}: points={v['points']} edges={v['edges']} deg_mean={v['deg_mean']:.2f} comps={c['components']} largest={c['largest_frac']:.3f} econn={v['electrode_connected_frac']:.3f} sel={v['deploy']['selected']}")

    # Auto-download ZIP in Colab
    if zip_path and args.auto_download and in_colab():
        print("[colab] downloading zip…")
        colab_download(zip_path)
    elif zip_path and args.auto_download and (not in_colab()):
        print("[note] auto_download requested but not running in Colab. Zip at:", os.path.abspath(zip_path))

if __name__ == "__main__":
    main()
