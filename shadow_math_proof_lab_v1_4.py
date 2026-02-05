#!/usr/bin/env python3
# ==============================================================
# Shadow Math Proof Lab v1.4 — Audit-grade Dimension Certificate
# ==============================================================
# Single-file, numpy + matplotlib only.
#
# What it does (deterministic):
#   (A) Demonstrates single-view non-injectivity (same 2D shadow, different 3D).
#   (B) Reconstructs 3D from >=2 calibrated views via linear triangulation (DLT).
#   (C) Computes Gram-eigen dimension certificate from reconstructed distances:
#         - rho = λ3/λ2  (out-of-plane eigen ratio)
#         - eta = λ3/sum(λ+) (out-of-plane energy share)
#         - k_eff (energy effective dimension)
#         - stress2 (MDS stress squared for k=2)
#   (D) Runs a 2D-in-3D control and an ablation that projects recon->best plane.
#   (E) Writes:
#         - JSON report
#         - PNG figures (eigs + 3D scatters)
#         - LaTeX report embedding the JSON (compile-safe pdfLaTeX)
#
# Usage:
#   python shadow_math_proof_lab_v1_4.py --outdir out_shadow --autorun
#
# Notes:
#   - "pixel units" are arbitrary; projection uses focal_px and principal point at (0,0).
#   - This is a mathematical/geometry certificate, not a physics claim.
#
# ==============================================================
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- utils -----------------------------
def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def robust_mean(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def robust_max(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.max(x))


def stable_sort_desc(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals).reshape(-1)
    return vals[np.argsort(-vals)]


def percent(x: float) -> float:
    return 100.0 * x


# ----------------------------- camera model -----------------------------
@dataclass
class Camera:
    # Extrinsics: X_cam
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3,
    # Intrinsics (calibrated-ish)
    focal_px: float

    def P(self) -> np.ndarray:
        """3x4 projection matrix P = K [R|t] with K = [[f,0,0],[0,f,0],[0,0,1]]."""
        f = float(self.focal_px)
        K = np.array([[f, 0.0, 0.0],
                      [0.0, f, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        Rt = np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)
        return K @ Rt

    def project(self, Xw: np.ndarray) -> np.ndarray:
        """Project world points (N,3) to pixels (N,2)."""
        Xw = np.asarray(Xw, dtype=np.float64)
        N = Xw.shape[0]
        Xh = np.concatenate([Xw, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
        xh = (self.P() @ Xh.T).T  # (N,3)
        u = xh[:, 0] / xh[:, 2]
        v = xh[:, 1] / xh[:, 2]
        return np.stack([u, v], axis=1)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct world->camera rotation R and translation t such that:
      X_cam = R X_world + t
    with camera at eye, looking at target, z-forward convention.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(x, z)

    # Camera coordinate axes as rows (world->cam)
    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    return R, t


# ----------------------------- triangulation -----------------------------
def triangulate_dlt(Ps: List[np.ndarray], xs: List[np.ndarray]) -> np.ndarray:
    """
    Linear triangulation by DLT.
    Inputs:
      Ps: list of (3,4) projection matrices
      xs: list of (2,) pixel coords corresponding to same point across views
    Output:
      X: (3,) world coordinate
    """
    A_rows = []
    for P, x in zip(Ps, xs):
        u, v = float(x[0]), float(x[1])
        A_rows.append(u * P[2, :] - P[0, :])
        A_rows.append(v * P[2, :] - P[1, :])
    A = np.stack(A_rows, axis=0)  # (2m,4)
    # Solve A X = 0 via SVD
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    Xh = Vt[-1, :]  # smallest singular value
    if abs(Xh[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    X = Xh[:3] / Xh[3]
    return X.astype(np.float64)


def reprojection_errors(Ps: List[np.ndarray], X: np.ndarray, xs: List[np.ndarray]) -> np.ndarray:
    """Return per-view reprojection error in pixel units."""
    X = np.asarray(X, dtype=np.float64).reshape(3)
    Xh = np.concatenate([X, np.array([1.0])], axis=0)  # (4,)
    errs = []
    for P, x in zip(Ps, xs):
        xh = P @ Xh
        u = xh[0] / xh[2]
        v = xh[1] / xh[2]
        e = math.sqrt((u - float(x[0]))**2 + (v - float(x[1]))**2)
        errs.append(e)
    return np.asarray(errs, dtype=np.float64)


# ----------------------------- distance / Gram / MDS -----------------------------
def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    G = X @ X.T
    diag = np.diag(G).reshape(-1, 1)
    D2 = diag + diag.T - 2.0 * G
    D2[D2 < 0] = 0.0
    return D2


def double_center_gram(D2: np.ndarray) -> np.ndarray:
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
    B = -0.5 * (J @ D2 @ J)
    # symmetrize for numerical stability
    return 0.5 * (B + B.T)


def gram_eigs(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(B)
    # descending
    idx = np.argsort(-w)
    w = w[idx]
    V = V[:, idx]
    return w, V


def mds_embedding_from_gram(B: np.ndarray, k: int) -> np.ndarray:
    w, V = gram_eigs(B)
    w_pos = np.maximum(w, 0.0)
    k = int(k)
    k = max(1, min(k, B.shape[0]))
    Wk = np.diag(np.sqrt(w_pos[:k]))
    Xk = V[:, :k] @ Wk
    return Xk


def stress2(D: np.ndarray, Dhat: np.ndarray) -> float:
    num = np.linalg.norm(D - Dhat, ord="fro")
    den = np.linalg.norm(D, ord="fro") + 1e-12
    return float((num / den) ** 2)


def eff_dim_from_eigs(w: np.ndarray, energy_eps: float) -> int:
    w_pos = np.maximum(w, 0.0)
    tot = float(np.sum(w_pos)) + 1e-12
    tail = np.cumsum(w_pos[::-1])[::-1]  # tail[i] = sum_{j>=i} w_pos[j]
    for k in range(1, len(w_pos) + 1):
        rem = float(np.sum(w_pos[k:]))
        if rem / tot <= float(energy_eps):
            return k
    return len(w_pos)


def rho_eta(w: np.ndarray) -> Tuple[float, float]:
    w_pos = np.maximum(w, 0.0)
    if len(w_pos) < 3:
        return 0.0, 0.0
    lam2 = float(w_pos[1])
    lam3 = float(w_pos[2])
    rho = lam3 / (lam2 + 1e-12) if lam2 > 0 else 0.0
    eta = lam3 / (float(np.sum(w_pos)) + 1e-12)
    return float(rho), float(eta)


def best_fit_plane_projection(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Project points to their best-fit plane (SVD).
    Returns X_proj and diagnostics including RMS orthogonal residual.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0, keepdims=True)
    Y = X - mu
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    # plane basis: first two right-singular vectors
    B = Vt[:2, :].T  # (3,2)
    # project
    Z = Y @ B        # (N,2)
    Yp = Z @ B.T     # (N,3)
    Xp = Yp + mu
    # residuals
    resid = Y - Yp
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return Xp, {"plane_rms": rms}


# ----------------------------- plotting -----------------------------
def plot_eigs(w: np.ndarray, title: str, outpath: str, top_k: int = 30) -> None:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    k = min(int(top_k), w.size)
    xs = np.arange(k)
    ys = w[:k]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_3d_scatter(true_X: np.ndarray, recon_X: np.ndarray, title: str, outpath: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(true_X[:, 0], true_X[:, 1], true_X[:, 2], s=18, label="true")
    ax.scatter(recon_X[:, 0], recon_X[:, 1], recon_X[:, 2], s=18, label="recon")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_3d_scatter_three(a: np.ndarray, b: np.ndarray, c: np.ndarray, labels: Tuple[str, str, str], title: str, outpath: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=18, label=labels[0])
    ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=18, label=labels[1])
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=18, label=labels[2])
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ----------------------------- LaTeX generator -----------------------------
def make_latex(report_json_text: str) -> str:
    """
    Produce a compile-safe, self-contained pdfLaTeX document embedding the JSON verbatim.
    """
    return r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{microtype}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fvextra}
\DefineVerbatimEnvironment{JSON}{Verbatim}{fontsize=\small,frame=single,breaklines=true,breakanywhere=true}

\definecolor{octaGray}{RGB}{50,50,50}
\definecolor{octaBlue}{RGB}{70,120,255}
\hypersetup{colorlinks=true, linkcolor=octaBlue, urlcolor=octaBlue, citecolor=octaBlue}

\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}

\newcommand{\OCTAResearch}{\textsc{OCTA Research}}
\newcommand{\Tagline}{\textit{The Environment becomes your weapon.}}

\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}{Lemma}[section]
\newtheorem{corollary}{Corollary}[section]
\theoremstyle{remark}
\newtheorem{remark}{Remark}[section]

\title{\vspace{-1.2em}{\Huge \OCTAResearch}\\[-0.2em]{\large \color{octaGray}Technical Note}\\[0.8em]
{\LARGE \textbf{Shadow Math / Calculus}}\\[-0.1em]
{\large \color{octaGray}Projection--Lift Geometry and Audit-Grade Dimension Certificates}\\[0.2em]
{\small \color{octaGray}\Tagline}}
\author{\OCTAResearch}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This document is auto-generated by \texttt{shadow\_math\_proof\_lab\_v1\_4.py}.
It embeds the complete JSON audit report verbatim (Appendix A).
\end{abstract}

\tableofcontents
\newpage

\section{Appendix A: Embedded Audit Artifact (JSON)}
\begin{JSON}
""" + report_json_text + r"""
\end{JSON}

\end{document}
""".lstrip()


# ----------------------------- main experiment -----------------------------
@dataclass
class Settings:
    seed: int = 7
    N_points: int = 120
    noise_sigma_px: float = 0.7
    views: int = 3
    focal_px: float = 800.0
    baseline: float = 0.7
    energy_eps: float = 0.002
    rho_tau: float = 0.01
    eta_tau: float = 0.005


def build_cameras(cfg: Settings) -> List[Camera]:
    """
    Three cameras around the origin, looking at a target in front.
    """
    target = np.array([0.0, 0.0, 4.0], dtype=np.float64)
    # Baseline on x plus some mild y offsets for non-degeneracy
    eyes = [
        np.array([0.0, 0.0, 0.0]),
        np.array([cfg.baseline, 0.1, 0.0]),
        np.array([-0.2, -0.15, 0.0]),
    ][: int(cfg.views)]
    cams = []
    for eye in eyes:
        R, t = look_at(eye=eye, target=target)
        cams.append(Camera(R=R, t=t, focal_px=float(cfg.focal_px)))
    return cams


def gen_3d_world(rng: np.random.Generator, N: int) -> np.ndarray:
    """
    3D points occupying volume around z ~ [2.5, 5.5]
    """
    x = rng.uniform(-0.7, 0.7, size=N)
    y = rng.uniform(-0.45, 0.45, size=N)
    z = rng.uniform(2.5, 5.6, size=N)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def gen_2d_in_3d_control(rng: np.random.Generator, N: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Points that lie (exactly) on a plane in 3D.
    """
    # Choose a random plane normal and a point
    n = rng.normal(size=3)
    n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.array([0.0, 0.0, 3.5], dtype=np.float64)

    # Create orthonormal basis (b1,b2) spanning the plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, n))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    b1 = np.cross(n, tmp)
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    b2 = np.cross(n, b1)

    u = rng.uniform(-0.7, 0.7, size=N)
    v = rng.uniform(-0.45, 0.45, size=N)
    X = p0 + np.outer(u, b1) + np.outer(v, b2)

    return X.astype(np.float64), {"plane_normal": n.tolist(), "plane_point": p0.tolist()}


def observe(cams: List[Camera], X: np.ndarray, rng: np.random.Generator, sigma_px: float) -> List[np.ndarray]:
    xs = []
    for cam in cams:
        x = cam.project(X)
        x = x + rng.normal(scale=float(sigma_px), size=x.shape)
        xs.append(x.astype(np.float64))
    return xs


def reconstruct(cams: List[Camera], xs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    Ps = [cam.P() for cam in cams]
    N = xs[0].shape[0]
    Xhat = np.zeros((N, 3), dtype=np.float64)
    reproj_err = np.zeros((N, len(Ps)), dtype=np.float64)
    for i in range(N):
        x_views = [x[i] for x in xs]
        X = triangulate_dlt(Ps, x_views)
        Xhat[i] = X
        reproj_err[i] = reprojection_errors(Ps, X, x_views)
    return Xhat, reproj_err


def cert_from_points(X: np.ndarray, energy_eps: float) -> Dict[str, object]:
    """
    Compute Gram-eigen diagnostics and stress2.
    """
    D2 = pairwise_sq_dists(X)
    D = np.sqrt(D2)
    B = double_center_gram(D2)
    w, _ = gram_eigs(B)

    rho, eta = rho_eta(w)
    k_eff = eff_dim_from_eigs(w, energy_eps=float(energy_eps))

    X2 = mds_embedding_from_gram(B, 2)
    Dhat2 = np.sqrt(pairwise_sq_dists(X2))
    s2 = stress2(D, Dhat2)

    return {
        "eigvals": w.tolist(),
        "rho": float(rho),
        "eta": float(eta),
        "k_eff": int(k_eff),
        "stress2_k2": float(s2),
    }


def is_planar(cert: Dict[str, object], rho_tau: float, eta_tau: float) -> bool:
    return (float(cert["rho"]) <= float(rho_tau)) and (float(cert["eta"]) <= float(eta_tau))


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--outdir", type=str, default="out_shadow_math_proof_lab_v1_4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--sigma_px", type=float, default=0.7)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--focal_px", type=float, default=800.0)
    ap.add_argument("--baseline", type=float, default=0.7)
    ap.add_argument("--energy_eps", type=float, default=0.002)
    ap.add_argument("--rho_tau", type=float, default=0.01)
    ap.add_argument("--eta_tau", type=float, default=0.005)
    ap.add_argument("--autorun", action="store_true", help="Run end-to-end and write outputs.")
    args, _unknown = ap.parse_known_args()  # Jupyter-safe

    cfg = Settings(
        seed=int(args.seed),
        N_points=int(args.N),
        noise_sigma_px=float(args.sigma_px),
        views=int(args.views),
        focal_px=float(args.focal_px),
        baseline=float(args.baseline),
        energy_eps=float(args.energy_eps),
        rho_tau=float(args.rho_tau),
        eta_tau=float(args.eta_tau),
    )

    outdir = ensure_dir(str(args.outdir))
    figs_dir = ensure_dir(os.path.join(outdir, "figs"))

    rng = set_seed(cfg.seed)
    cams = build_cameras(cfg)

    # ----------------------------------------------------------
    # (A) Single-view non-injectivity witness
    # ----------------------------------------------------------
    # Choose a random pixel (u,v) and construct two points on the same ray for cam0.
    cam0 = cams[0]
    u, v = rng.uniform(-200, 200), rng.uniform(-150, 150)
    # In camera coords: point (X,Y,Z) projects to (fX/Z,fY/Z)=(u,v)
    # So X=(u/f)Z, Y=(v/f)Z. Pick two depths:
    Z1, Z2 = 3.0, 5.0
    f = cfg.focal_px
    Xc1 = np.array([(u/f)*Z1, (v/f)*Z1, Z1])
    Xc2 = np.array([(u/f)*Z2, (v/f)*Z2, Z2])

    # Convert cam coords to world coords: X_cam = R X_world + t  => X_world = R^T (X_cam - t)
    def cam_to_world(cam: Camera, Xc: np.ndarray) -> np.ndarray:
        return cam.R.T @ (Xc - cam.t)

    Xw1 = cam_to_world(cam0, Xc1)
    Xw2 = cam_to_world(cam0, Xc2)

    p1 = cam0.project(Xw1.reshape(1, 3))[0]
    p2 = cam0.project(Xw2.reshape(1, 3))[0]
    noninj_pixel_dist = float(np.linalg.norm(p1 - p2))

    # ----------------------------------------------------------
    # (3D world)
    # ----------------------------------------------------------
    X3_true = gen_3d_world(rng, cfg.N_points)
    xs3 = observe(cams, X3_true, rng, cfg.noise_sigma_px)
    X3_recon, reproj3 = reconstruct(cams, xs3)
    cert3 = cert_from_points(X3_recon, cfg.energy_eps)
    planar3 = is_planar(cert3, cfg.rho_tau, cfg.eta_tau)

    # ----------------------------------------------------------
    # (2D-in-3D control)
    # ----------------------------------------------------------
    X2_true, plane_meta = gen_2d_in_3d_control(rng, cfg.N_points)
    xs2 = observe(cams, X2_true, rng, cfg.noise_sigma_px)
    X2_recon, reproj2 = reconstruct(cams, xs2)
    cert2 = cert_from_points(X2_recon, cfg.energy_eps)
    planar2 = is_planar(cert2, cfg.rho_tau, cfg.eta_tau)

    # Ablation: project recon->best plane
    X2_proj, plane_diag = best_fit_plane_projection(X2_recon)
    cert2p = cert_from_points(X2_proj, cfg.energy_eps)
    planar2p = is_planar(cert2p, cfg.rho_tau, cfg.eta_tau)

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------
    plot_eigs(np.array(cert3["eigvals"]), "3D reconstructed: top Gram eigenvalues (B)",
              os.path.join(figs_dir, "eig_3d_reconstructed.png"))
    plot_eigs(np.array(cert2["eigvals"]), "2D reconstructed: top Gram eigenvalues (B)",
              os.path.join(figs_dir, "eig_2d_reconstructed.png"))
    plot_eigs(np.array(cert2p["eigvals"]), "2D recon projected-to-plane: top Gram eigenvalues (B)",
              os.path.join(figs_dir, "eig_2d_projected_to_plane.png"))

    plot_3d_scatter(X3_true, X3_recon, "3D world: true vs reconstructed",
                    os.path.join(figs_dir, "scatter_3d_true_vs_recon.png"))
    plot_3d_scatter_three(X2_true, X2_recon, X2_proj,
                          labels=("true 2D-in-3D", "recon", "recon->plane"),
                          title="2D control: true vs reconstructed vs projected-to-plane",
                          outpath=os.path.join(figs_dir, "scatter_2d_true_recon_plane.png"))

    # ----------------------------------------------------------
    # Report JSON
    # ----------------------------------------------------------
    report = {
        "shadow_math_proof_lab": "v1.4",
        "settings": asdict(cfg),
        "partA_single_view_noninjective": {
            "pixel_uv": [float(u), float(v)],
            "Z1": float(Z1),
            "Z2": float(Z2),
            "X_world_1": Xw1.tolist(),
            "X_world_2": Xw2.tolist(),
            "proj_cam0_1": p1.tolist(),
            "proj_cam0_2": p2.tolist(),
            "pixel_dist": noninj_pixel_dist,
        },
        "reprojection": {
            "3d_mean": robust_mean(reproj3),
            "3d_median": robust_median(reproj3),
            "3d_max": robust_max(reproj3),
            "2d_mean": robust_mean(reproj2),
            "2d_median": robust_median(reproj2),
            "2d_max": robust_max(reproj2),
        },
        "certificates": {
            "3d_reconstructed": {
                **cert3,
                "planar": bool(planar3),
            },
            "2d_reconstructed": {
                **cert2,
                "planar": bool(planar2),
                "plane_meta": plane_meta,
            },
            "2d_recon_projected_to_plane": {
                **cert2p,
                "planar": bool(planar2p),
                "projection_diag": plane_diag,
            },
        },
        "notes": {
            "meaning": "Mathematical dimension certificate from Gram eigenstructure; not a physical spacetime claim.",
            "planarity_rule": {"rho_tau": cfg.rho_tau, "eta_tau": cfg.eta_tau},
            "k_eff_rule": {"energy_eps": cfg.energy_eps},
        },
    }

    report_path = os.path.join(outdir, "shadow_proof_report_v1_4.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ----------------------------------------------------------
    # LaTeX (self-contained, embeds report JSON verbatim)
    # ----------------------------------------------------------
    latex_text = make_latex(Path(report_path).read_text(encoding="utf-8"))
    latex_path = os.path.join(outdir, "OCTA_Shadow_Math_Audit_v1_4.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    # ----------------------------------------------------------
    # Console summary (matches the style you used)
    # ----------------------------------------------------------
    print("==============================================================")
    print("Shadow Math Proof Lab v1.4 — Audit-grade Dimension Certificate")
    print("==============================================================")
    print(f"(A) Single-view non-injectivity pixel dist: {noninj_pixel_dist:.6f} (should be ~0)")
    print("")
    print(f"[3D] reproj mean/med/max px: {report['reprojection']['3d_mean']:.3f}/{report['reprojection']['3d_median']:.3f}/{report['reprojection']['3d_max']:.3f}")
    print(f"[3D] rho={cert3['rho']:.6g} | eta={cert3['eta']:.6g} | k_eff={cert3['k_eff']} | stress2={cert3['stress2_k2']:.6g}")
    print(f"[3D] CERT (planar?): {planar3}  => EXPECT non-planar => {'PASS' if (not planar3) else 'FAIL'}")
    print("")
    print(f"[2D] reproj mean/med/max px: {report['reprojection']['2d_mean']:.3f}/{report['reprojection']['2d_median']:.3f}/{report['reprojection']['2d_max']:.3f}")
    print(f"[2D] rho={cert2['rho']:.6g} | eta={cert2['eta']:.6g} | k_eff={cert2['k_eff']} | stress2={cert2['stress2_k2']:.6g}")
    print(f"[2D] CERT (planar?): {planar2}  => EXPECT near-planar => {'PASS' if planar2 else 'FAIL'}")
    print("")
    print("[Ablation] 2D recon projected-to-plane should become 'more planar' (lambda3 collapses, stress2 drops).")
    print(f"          2D_proj rho={cert2p['rho']:.6g} | eta={cert2p['eta']:.6g} | k_eff={cert2p['k_eff']} | stress2={cert2p['stress2_k2']:.6g} | plane_rms={plane_diag['plane_rms']:.6g}")
    print("")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {latex_path}")
    print(f"Figures: {figs_dir}")

if __name__ == "__main__":
    main()
