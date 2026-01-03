import math
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For Colab / Jupyter HTML display
try:
    from IPython.display import HTML, IFrame, display
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class SpatialAIConfig:
    # Grid / domain
    nx: int = 96
    ny: int = 72
    Lx: float = 10.0
    Ly: float = 8.0

    # Physics
    kappa_base: float = 1.0
    kappa_route: float = 4.5
    heat_source_strength: float = 5.0
    heat_sigma: float = 1.5

    # Persistent source / decay
    source_decay: float = 0.05    # relaxation rate toward T_env
    T_env: float = 0.0            # ambient baseline

    # Nodes / attractor
    n_nodes_x: int = 3
    n_nodes_y: int = 3
    attractor_pos: Tuple[float, float] = (5.0, 4.0)

    # Time integration
    t_final: float = 0.24
    dt: float = None  # if None, CFL-safe
    reconfig_step: int = 200

    # Node dynamics
    kuramoto_K: float = 0.6
    whisper_coupling: float = 1.5
    mu_noise_std: float = 0.015

    # TFS routing rectangles (x_min, x_max, y_min, y_max) in domain coords
    route_rects: Tuple[Tuple[float, float, float, float], ...] = (
        (2.0, 3.0, 2.5, 3.5),
        (4.0, 5.0, 3.5, 4.5),
        (8.0, 9.0, 5.5, 6.5),
    )


# ---------------------------------------------------------------------
# Utility: grid / fields
# ---------------------------------------------------------------------
def make_grid(cfg: SpatialAIConfig):
    x = np.linspace(0.0, cfg.Lx, cfg.nx)
    y = np.linspace(0.0, cfg.Ly, cfg.ny)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return X, Y, dx, dy


def make_initial_T(cfg: SpatialAIConfig, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Initial Gaussian hump around attractor."""
    ax, ay = cfg.attractor_pos
    r2 = (X - ax) ** 2 + (Y - ay) ** 2
    T0 = cfg.heat_source_strength * np.exp(-0.5 * r2 / (cfg.heat_sigma ** 2))
    return T0


def make_source(cfg: SpatialAIConfig, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Persistent heat source centered on attractor.
    This is pumped every time-step.
    """
    ax, ay = cfg.attractor_pos
    r2 = (X - ax) ** 2 + (Y - ay) ** 2
    base = np.exp(-0.5 * r2 / (cfg.heat_sigma ** 2))
    S = cfg.heat_source_strength * base
    return S


def make_kappa(cfg: SpatialAIConfig,
               X: np.ndarray,
               Y: np.ndarray,
               kappa_route: float) -> np.ndarray:
    """Base conductivity + boosted in route rectangles."""
    kappa = np.full_like(X, cfg.kappa_base, dtype=float)
    for (xmin, xmax, ymin, ymax) in cfg.route_rects:
        mask = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
        kappa[mask] = cfg.kappa_base + kappa_route
    return kappa


def step_heat(T: np.ndarray,
              kappa: np.ndarray,
              dx: float,
              dy: float,
              dt: float,
              source: np.ndarray = None,
              decay: float = 0.0,
              T_env: float = 0.0) -> np.ndarray:
    """
    Single explicit step for:
       dT/dt = ∇·(kappa ∇T) - decay * (T - T_env) + source
    with spatially varying kappa.
    """
    Tx = np.zeros_like(T)
    Ty = np.zeros_like(T)

    # x-second derivative
    Tx[:, 1:-1] = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / (dx * dx)

    # y-second derivative
    Ty[1:-1, :] = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / (dy * dy)

    lapT = Tx + Ty

    RHS = kappa * lapT - decay * (T - T_env)
    if source is not None:
        RHS = RHS + source

    T_new = T + dt * RHS

    # Neumann BCs (zero gradient)
    T_new[:, 0] = T_new[:, 1]
    T_new[:, -1] = T_new[:, -2]
    T_new[0, :] = T_new[1, :]
    T_new[-1, :] = T_new[-2, :]
    return T_new


# ---------------------------------------------------------------------
# Nodes: positions, Kuramoto, whispers
# ---------------------------------------------------------------------
def init_nodes(cfg: SpatialAIConfig):
    xs = np.linspace(cfg.Lx * 0.2, cfg.Lx * 0.8, cfg.n_nodes_x)
    ys = np.linspace(cfg.Ly * 0.2, cfg.Ly * 0.8, cfg.n_nodes_y)
    Xn, Yn = np.meshgrid(xs, ys)
    node_pos = np.column_stack([Xn.ravel(), Yn.ravel()])  # (N, 2)

    N = node_pos.shape[0]
    # Fully connected graph minus self-loops
    A = np.ones((N, N), dtype=float) - np.eye(N)

    # Baseline structural coherence (LCF prior) – tuned near 0.963
    C_S_base = 1.0 - 1.0 / (3 * N)  # ~0.963 for N=9

    return node_pos, A, C_S_base


def sample_heat_at_nodes(T: np.ndarray,
                         cfg: SpatialAIConfig,
                         node_pos: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sampling of T at node positions."""
    ny, nx = T.shape
    xs = (node_pos[:, 0] / cfg.Lx) * (nx - 1)
    ys = (node_pos[:, 1] / cfg.Ly) * (ny - 1)
    ix = np.clip(xs.round().astype(int), 0, nx - 1)
    iy = np.clip(ys.round().astype(int), 0, ny - 1)
    return T[iy, ix]


def step_kuramoto(theta: np.ndarray,
                  omega: np.ndarray,
                  K: float,
                  A: np.ndarray,
                  dt: float) -> np.ndarray:
    """Kuramoto update on a fixed adjacency matrix."""
    N = theta.size
    theta_diff = theta.reshape((N, 1)) - theta.reshape((1, N))
    coupling = (A * np.sin(theta_diff)).sum(axis=1) / np.maximum(A.sum(axis=1), 1.0)
    return theta + dt * (omega + K * coupling)


def update_whisper_beliefs(mu: np.ndarray,
                           heat_at_nodes: np.ndarray,
                           A: np.ndarray,
                           cfg: SpatialAIConfig,
                           rng: np.random.Generator) -> np.ndarray:
    """WGC-style update of node beliefs."""
    N = mu.size
    deg = np.maximum(A.sum(axis=1), 1.0)
    neighbor_mean = (A @ mu) / deg

    h = heat_at_nodes
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)

    # Soften the drive a bit to avoid saturating at tanh(·) ≈ 1
    drive = 0.5 * h_norm + 0.5 * neighbor_mean
    raw = cfg.whisper_coupling * drive
    noise = rng.normal(0.0, cfg.mu_noise_std, size=N)

    mu_new = np.tanh(raw + noise)
    return mu_new


# ---------------------------------------------------------------------
# Coherence metrics
# ---------------------------------------------------------------------
def field_entropy(T: np.ndarray) -> float:
    """Shannon entropy of normalized positive field."""
    P = np.maximum(T, 0.0)
    s = P.sum()
    if s <= 0.0:
        return 0.0
    P /= s
    P = P[P > 0]
    H = -float((P * np.log(P)).sum())
    return H


def normalized_entropy(T: np.ndarray) -> float:
    """Entropy normalized by log(#cells)."""
    ny, nx = T.shape
    H = field_entropy(T)
    H_max = math.log(nx * ny)
    return H / (H_max + 1e-9)


def intent_coherence(mu: np.ndarray,
                     heat_at_nodes: np.ndarray) -> float:
    """Cosine similarity between beliefs and local heat drive, mapped to [0,1]."""
    v1 = mu - mu.mean()
    v2 = heat_at_nodes - heat_at_nodes.mean()
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.5
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return 0.5 * (cos + 1.0)  # [-1,1] -> [0,1]


def dynamic_coherence(theta: np.ndarray) -> float:
    """Kuramoto order parameter magnitude."""
    z = np.exp(1j * theta)
    R = np.abs(z.mean())
    return float(R)


def heat_radius(T: np.ndarray,
                cfg: SpatialAIConfig,
                X: np.ndarray,
                Y: np.ndarray) -> float:
    """Energy-weighted radial spread around attractor."""
    ax, ay = cfg.attractor_pos
    r2 = (X - ax) ** 2 + (Y - ay) ** 2
    w = np.maximum(T, 0.0)
    s = w.sum()
    if s <= 0.0:
        return 0.0
    r2_mean = float((w * r2).sum() / s)
    return math.sqrt(max(r2_mean, 0.0))


def classify_mode_series(C_ext: np.ndarray,
                         reconfig_step: int) -> List[str]:
    """
    Relative classifier: drift / nominal / supercoherent / alert.
    Thresholds adapt to each run's own baseline statistics, using
    the pre-reconfig window as 'init' regime.
    """
    n = C_ext.size
    if n == 0:
        return []

    # Use pre-reconfig window as baseline for thresholds
    q = max(1, min(reconfig_step, n) // 2)
    base_mean = float(np.mean(C_ext[:q]))
    base_std = float(np.std(C_ext[:q]) + 1e-9)

    super_thr = base_mean + 0.75 * base_std
    drift_thr = base_mean - 0.75 * base_std
    alert_drop = 0.06

    modes: List[str] = []
    prev = C_ext[0]
    for t in range(n):
        c = C_ext[t]
        if c >= super_thr:
            mode = "supercoherent"
        elif c <= drift_thr:
            mode = "drift"
        else:
            mode = "nominal"
        if prev - c > alert_drop:
            mode = "alert"
        modes.append(mode)
        prev = c

    return modes


# ---------------------------------------------------------------------
# TFS reconfiguration
# ---------------------------------------------------------------------
def tfs_reconfigure(cfg: SpatialAIConfig) -> SpatialAIConfig:
    """
    TFS reconfiguration: narrow and pull all route rectangles
    toward the attractor, boost route kappa strongly, reduce
    background kappa more aggressively, tighten the heat source,
    and amplify its strength.
    """
    ax, ay = cfg.attractor_pos
    new_rects = []
    for (xmin, xmax, ymin, ymax) in cfg.route_rects:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        w = (xmax - xmin) * 0.6
        h = (ymax - ymin) * 0.6

        # pull centers 50 % toward the attractor
        cx = cx + 0.5 * (ax - cx)
        cy = cy + 0.5 * (ay - cy)

        xmin_new = max(0.0, cx - 0.5 * w)
        xmax_new = min(cfg.Lx, cx + 0.5 * w)
        ymin_new = max(0.0, cy - 0.5 * h)
        ymax_new = min(cfg.Ly, cy + 0.5 * h)
        new_rects.append((xmin_new, xmax_new, ymin_new, ymax_new))

    # Stronger contrast after reconfig
    new_params = {
        **asdict(cfg),
        "route_rects": tuple(new_rects),
        "kappa_route": cfg.kappa_route * 1.6,
        "kappa_base": cfg.kappa_base * 0.6,
        "heat_source_strength": cfg.heat_source_strength * 1.4,
        "heat_sigma": cfg.heat_sigma * 0.75,
        "source_decay": cfg.source_decay * 0.7,
    }
    new_cfg = SpatialAIConfig(**new_params)
    return new_cfg


# ---------------------------------------------------------------------
# Core runner with TFS and coherence tracking
# ---------------------------------------------------------------------
def run_spatial_ai(cfg: SpatialAIConfig,
                   seed: int = 17,
                   run_id: str = "demo") -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    X, Y, dx, dy = make_grid(cfg)
    T = make_initial_T(cfg, X, Y)
    T0 = T.copy()

    # Baseline heat radius for focus metric
    r0 = heat_radius(T0, cfg, X, Y) + 1e-9

    kappa = make_kappa(cfg, X, Y, cfg.kappa_route)
    source = make_source(cfg, X, Y)

    # CFL-safe dt if not provided (for diffusion part only)
    if cfg.dt is None:
        kmax = cfg.kappa_base + cfg.kappa_route
        dt_cfl = 0.24 * min(dx, dy) ** 2 / (kmax + 1e-9)
        dt = 0.95 * dt_cfl
    else:
        dt = cfg.dt

    n_steps = int(cfg.t_final / dt)
    print(f"[INFO] CFL-safe dt ≈ {dt:.4e}")
    print(f"[INFO] Steps: {n_steps}")

    # Nodes
    node_pos, A, C_S_base = init_nodes(cfg)
    N = node_pos.shape[0]
    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    omega0 = rng.normal(0.0, 0.3, size=N)
    mu = rng.normal(0.0, 0.1, size=N)

    # Time histories
    C_S_hist, C_D_hist = [], []
    C_int_hist, C_info_hist = [], []
    C_LCF_hist, C_ext_hist = [], []
    C_focus_hist = []
    mu_mean_hist = []
    heat_focus_hist = []

    T_pre = None
    T_post = None

    for t in range(n_steps):
        # Heat update with persistent source + decay
        T = step_heat(
            T,
            kappa,
            dx,
            dy,
            dt,
            source=source,
            decay=cfg.source_decay,
            T_env=cfg.T_env,
        )

        # sample heat at nodes & update node dynamics
        h_nodes = sample_heat_at_nodes(T, cfg, node_pos)
        theta = step_kuramoto(theta, omega0, cfg.kuramoto_K, A, dt)
        mu = update_whisper_beliefs(mu, h_nodes, A, cfg, rng)

        # coherence metrics
        C_S = C_S_base
        C_D = dynamic_coherence(theta)
        C_int = intent_coherence(mu, h_nodes)
        C_info = normalized_entropy(T)
        C_LCF = 1.0

        r = heat_radius(T, cfg, X, Y)
        heat_focus_hist.append(r)
        # Convert radius contraction into [0,1] focus score
        # r0 / r > 1 means contraction; cap at factor 2
        focus_score = np.clip(r0 / (r + 1e-9), 0.0, 2.0) / 2.0
        C_focus = float(focus_score)
        C_focus_hist.append(C_focus)

        # Unified external coherence:
        # structural + dynamic + intent + geometric focus
        C_ext = (
            0.30 * C_S +
            0.35 * C_D +
            0.20 * C_int +
            0.15 * C_focus
        )

        C_S_hist.append(C_S)
        C_D_hist.append(C_D)
        C_int_hist.append(C_int)
        C_info_hist.append(C_info)
        C_LCF_hist.append(C_LCF)
        C_ext_hist.append(C_ext)
        mu_mean_hist.append(float(mu.mean()))

        if t == cfg.reconfig_step - 1:
            T_pre = T.copy()
        if t == n_steps - 1:
            T_post = T.copy()

        # TFS reconfiguration
        if t == cfg.reconfig_step:
            print(f"[INFO] Mode transition at step {t:4d}: init → nominal")
            print(f"[INFO] Applying TFS reconfiguration at step {t}")
            cfg = tfs_reconfigure(cfg)
            kappa = make_kappa(cfg, X, Y, cfg.kappa_route)
            source = make_source(cfg, X, Y)  # updated, tighter source

    # Convert histories
    C_S_hist = np.array(C_S_hist)
    C_D_hist = np.array(C_D_hist)
    C_int_hist = np.array(C_int_hist)
    C_info_hist = np.array(C_info_hist)
    C_LCF_hist = np.array(C_LCF_hist)
    C_ext_hist = np.array(C_ext_hist)
    C_focus_hist = np.array(C_focus_hist)
    mu_mean_hist = np.array(mu_mean_hist)
    heat_focus_hist = np.array(heat_focus_hist)

    # Mode classification
    modes = classify_mode_series(C_ext_hist, cfg.reconfig_step)
    modes_arr = np.array(modes, dtype=object)

    def frac_mode(name: str) -> float:
        return float(np.mean(modes_arr == name)) if modes_arr.size > 0 else 0.0

    frac_super = frac_mode("supercoherent")
    frac_nominal = frac_mode("nominal")
    frac_drift = frac_mode("drift")
    frac_alert = frac_mode("alert")

    # time to supercoherent AFTER reconfig
    idx_super = np.where(
        (modes_arr == "supercoherent") &
        (np.arange(len(modes_arr)) >= cfg.reconfig_step)
    )[0]
    t_to_super = float(idx_super[0] - cfg.reconfig_step) if idx_super.size > 0 else np.nan

    # pre/post coherence windows
    if n_steps > cfg.reconfig_step:
        C_ext_pre = float(C_ext_hist[:cfg.reconfig_step].mean())
        C_ext_post = float(C_ext_hist[cfg.reconfig_step:].mean())
    else:
        C_ext_pre = C_ext_post = float(C_ext_hist.mean())

    gain_reconfig = C_ext_post - C_ext_pre
    C_ext_mean = float(C_ext_hist.mean())
    C_ext_final = float(C_ext_hist[-1])

    # heat compaction: >1 means contraction after reconfig
    if T_pre is not None and T_post is not None:
        r_pre = heat_radius(T_pre, cfg, X, Y)
        r_post = heat_radius(T_post, cfg, X, Y) + 1e-9
        heat_compaction = float(r_pre / r_post)
    else:
        heat_compaction = np.nan

    delta_C_ext = C_ext_final - C_ext_mean

    return dict(
        cfg=cfg,
        X=X,
        Y=Y,
        T0=T0,
        T=T,
        kappa=kappa,
        node_pos=node_pos,
        theta_hist=theta,
        mu_hist=mu,
        C_S_hist=C_S_hist,
        C_D_hist=C_D_hist,
        C_int_hist=C_int_hist,
        C_info_hist=C_info_hist,
        C_LCF_hist=C_LCF_hist,
        C_ext_hist=C_ext_hist,
        C_focus_hist=C_focus_hist,
        mu_mean_hist=mu_mean_hist,
        heat_focus_hist=heat_focus_hist,
        modes=modes_arr,
        C_ext_mean=C_ext_mean,
        C_ext_final=C_ext_final,
        C_ext_pre=C_ext_pre,
        C_ext_post=C_ext_post,
        gain_reconfig=gain_reconfig,
        heat_compaction=heat_compaction,
        t_to_super=t_to_super,
        delta_C_ext=delta_C_ext,
        frac_supercoherent=frac_super,
        frac_nominal=frac_nominal,
        frac_drift=frac_drift,
        frac_alert=frac_alert,
        run_id=run_id,
        seed=seed,
        n_steps=n_steps,
        nx=cfg.nx,
        ny=cfg.ny,
    )


# ---------------------------------------------------------------------
# Plotting helpers (unchanged except for optional focus overlay)
# ---------------------------------------------------------------------
def plot_spatial_panels(res: Dict[str, Any], fname: str = "spatial_panels.png"):
    cfg = res["cfg"]
    X, Y = res["X"], res["Y"]
    T = res["T"]
    kappa = res["kappa"]
    node_pos = res["node_pos"]
    mu_final = res["mu_hist"]
    ax_attr, ay_attr = cfg.attractor_pos

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))

    im0 = axes[0].imshow(T, origin="lower",
                         extent=[0, cfg.Lx, 0, cfg.Ly])
    axes[0].set_title("Heat field (TFS routing)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(kappa, origin="lower",
                         extent=[0, cfg.Lx, 0, cfg.Ly])
    axes[1].set_title(r"$\kappa_{\mathrm{eff}}$ (routes)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    sc = axes[2].scatter(node_pos[:, 0], node_pos[:, 1],
                         c=mu_final, cmap="coolwarm",
                         vmin=-1.0, vmax=1.0, s=120, edgecolor="k")
    axes[2].scatter([ax_attr], [ay_attr],
                    marker="*", s=180, color="gold", edgecolor="k",
                    label="Attractor")
    axes[2].set_xlim(0, cfg.Lx)
    axes[2].set_ylim(0, cfg.Ly)
    axes[2].set_title("Node beliefs (WGC + attractor)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].legend(loc="upper left")
    fig.colorbar(sc, ax=axes[2], label="belief μ")

    fig.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.show()


def plot_core_coherence(res: Dict[str, Any],
                        fname: str = "core_coherence.png"):
    C_S = res["C_S_hist"]
    C_D = res["C_D_hist"]
    C_int = res["C_int_hist"]
    C_info = res["C_info_hist"]
    C_LCF = res["C_LCF_hist"]
    C_ext = res["C_ext_hist"]
    C_focus = res["C_focus_hist"]
    mu_mean = res["mu_mean_hist"]
    heat_focus = res["heat_focus_hist"]
    cfg = res["cfg"]
    n_steps = res["n_steps"]
    t = np.arange(n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

    axes[0].plot(t, C_S, label=r"$C_S$ (structural)")
    axes[0].plot(t, C_D, label=r"$C_D$ (dynamic)")
    axes[0].axvline(cfg.reconfig_step, color="k", linestyle="--",
                    label="TFS reconfig")
    axes[0].set_ylabel("Coherence")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Core coherence axes")
    axes[0].legend(loc="lower right")

    axes[1].plot(t, C_int, label=r"$C_{\mathrm{int}}$")
    axes[1].plot(t, C_info, label=r"$C_{\mathrm{info}}$")
    axes[1].plot(t, C_focus, label=r"$C_{\mathrm{focus}}$ (radius-based)")
    axes[1].axvline(cfg.reconfig_step, color="k", linestyle="--")
    axes[1].set_ylabel("Coherence")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")

    ax = axes[2]
    ax.plot(t, mu_mean, label="Whisper mean μ")
    ax.plot(t, C_LCF, label=r"$C_{\mathrm{LCF}}$")
    ax.plot(t, C_ext, label=r"$C_{\mathrm{ext}}$ (unified)")

    ax2 = ax.twinx()
    hf_norm = heat_focus / (heat_focus[0] + 1e-9)
    ax2.plot(t, hf_norm, ":", label="Heat radius / initial")

    ax.axvline(cfg.reconfig_step, color="k", linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Coherence / μ")
    ax.set_ylim(-0.25, 1.2)
    ax2.set_ylabel("Heat radius / initial")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------
def run_benchmark_grid(base_cfg: SpatialAIConfig) -> pd.DataFrame:
    seeds = [17, 18, 19, 20]
    k_route_values = [3.5, 4.5, 5.5]
    whisper_values = [1.5, 2.0]
    noise_values = [0.01, 0.02]

    records = []

    print("\n====================================")
    print(" BENCHMARK GRID (automated runs)")
    print("====================================")

    total_runs = len(seeds) * len(k_route_values) * len(whisper_values) * len(noise_values)
    run_index = 0

    for seed in seeds:
        for kr in k_route_values:
            for wc in whisper_values:
                for noise in noise_values:
                    run_id = f"run_{run_index:03d}"
                    print(f"[INFO] Benchmark run {run_index+1}/{total_runs} · "
                          f"id={run_id} seed={seed} κ_route={kr:.2f} wc={wc:.2f} noise={noise:.3f}")

                    cfg = SpatialAIConfig(
                        **{**asdict(base_cfg),
                           "kappa_route": kr,
                           "whisper_coupling": wc,
                           "mu_noise_std": noise}
                    )
                    res = run_spatial_ai(cfg, seed=seed, run_id=run_id)

                    records.append(dict(
                        run_id=run_id,
                        seed=seed,
                        n_steps=res["n_steps"],
                        nx=res["nx"],
                        ny=res["ny"],
                        kappa_route=kr,
                        whisper_coupling=wc,
                        mu_noise_std=noise,
                        kuramoto_K=cfg.kuramoto_K,
                        heat_source_strength=cfg.heat_source_strength,
                        C_ext_mean=res["C_ext_mean"],
                        C_ext_final=res["C_ext_final"],
                        C_S_mean=float(res["C_S_hist"].mean()),
                        C_D_mean=float(res["C_D_hist"].mean()),
                        C_int_mean=float(res["C_int_hist"].mean()),
                        C_info_mean=float(res["C_info_hist"].mean()),
                        C_focus_mean=float(res["C_focus_hist"].mean()),
                        C_LCF_mean=float(res["C_LCF_hist"].mean()),
                        frac_supercoherent=res["frac_supercoherent"],
                        frac_nominal=res["frac_nominal"],
                        frac_drift=res["frac_drift"],
                        frac_alert=res["frac_alert"],
                        C_ext_pre=res["C_ext_pre"],
                        C_ext_post=res["C_ext_post"],
                        gain_reconfig=res["gain_reconfig"],
                        heat_compaction=res["heat_compaction"],
                        t_to_super=res["t_to_super"],
                        delta_C_ext=res["delta_C_ext"],
                    ))

                    run_index += 1

    df = pd.DataFrame.from_records(records)
    return df


# ---------------------------------------------------------------------
# Benchmark plotting (unchanged except for extra column availability)
# ---------------------------------------------------------------------
def plot_benchmark_figures(df: pd.DataFrame):
    # lock-in speed vs routing strength (only runs that actually lock in)
    plt.figure(figsize=(4, 3))
    mask = ~df["t_to_super"].isna()
    if mask.any():
        plt.scatter(df.loc[mask, "kappa_route"],
                    df.loc[mask, "t_to_super"])
    plt.xlabel(r"$\kappa_{\mathrm{route}}$")
    plt.ylabel("Time to supercoherent after TFS (steps)")
    plt.title("Lock-in speed vs routing strength")
    ax = plt.gca()
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.savefig("lockin_vs_kappa.png", dpi=150, bbox_inches="tight")
    plt.show()

    # noise sensitivity – show plain numbers (no scientific offset)
    plt.figure(figsize=(4, 3))
    for kr in sorted(df["kappa_route"].unique()):
        subset = df[df["kappa_route"] == kr]
        means = subset.groupby("mu_noise_std")["C_ext_mean"].mean()
        plt.plot(means.index, means.values, marker="o",
                 label=fr"$\kappa_{{route}}={kr:.1f}$")
    plt.xlabel("μ noise std")
    plt.ylabel("Mean C_ext")
    plt.title("Noise sensitivity of unified coherence")
    plt.legend()
    ax = plt.gca()
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.savefig("noise_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()

    # TFS focus vs coherence gain
    plt.figure(figsize=(4, 3))
    plt.scatter(df["heat_compaction"], df["gain_reconfig"])
    plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("Heat compaction (pre/post radius)")
    plt.ylabel("Gain in mean C_ext after TFS")
    plt.title("TFS reconfig: focus vs coherence gain")
    ax = plt.gca()
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.savefig("focus_vs_gain.png", dpi=150, bbox_inches="tight")
    plt.show()

    # heatmap of C_ext_mean vs (kappa_route, whisper_coupling)
    pivot = df.pivot_table(index="whisper_coupling",
                           columns="kappa_route",
                           values="C_ext_mean",
                           aggfunc="mean")
    plt.figure(figsize=(4.5, 3.5))
    im = plt.imshow(pivot.values, origin="lower",
                    extent=[pivot.columns.min() - 0.5,
                            pivot.columns.max() + 0.5,
                            pivot.index.min() - 0.05,
                            pivot.index.max() + 0.05],
                    aspect="auto")
    plt.colorbar(im, label="Mean C_ext")
    plt.xlabel(r"$\kappa_{\mathrm{route}}$")
    plt.ylabel("Whisper coupling")
    plt.title("Mean C_ext vs κ_route × whisper_coupling")
    plt.savefig("heatmap_cext.png", dpi=150, bbox_inches="tight")
    plt.show()

    # histogram of C_ext_mean distribution
    plt.figure(figsize=(4, 3))
    plt.hist(df["C_ext_mean"], bins=12)
    plt.xlabel("C_ext (mean)")
    plt.ylabel("Count")
    plt.title("Benchmark distribution of unified coherence")
    ax = plt.gca()
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.savefig("hist_cext.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main execution (no HTML dashboard wiring here)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("====================================")
    print(" DEMO RUN (single Spatial AI system)")
    print("====================================")

    base_cfg = SpatialAIConfig()
    demo_res = run_spatial_ai(base_cfg, seed=17, run_id="demo")

    plot_spatial_panels(demo_res)
    plot_core_coherence(demo_res)

    df = run_benchmark_grid(base_cfg)

    print("\n====================================")
    print(" BENCHMARK ANALYSIS")
    print("====================================")
    print("\n=== Benchmark DataFrame head ===\n")
    if _HAS_IPY:
        display(df.head())
    else:
        print(df.head())

    key_cols = [
        "C_ext_mean",
        "C_ext_final",
        "C_S_mean",
        "C_D_mean",
        "C_int_mean",
        "C_info_mean",
        "C_focus_mean",
        "C_LCF_mean",
    ]
    print("\n=== Global descriptive stats (key metrics) ===\n")
    desc = df[key_cols].describe()
    if _HAS_IPY:
        display(desc)
    else:
        print(desc)

    impact_cols = ["gain_reconfig", "heat_compaction", "t_to_super", "delta_C_ext"]
    print("\n=== Reconfiguration impact metrics ===\n")
    impact = df[impact_cols].describe()
    if _HAS_IPY:
        display(impact)
    else:
        print(impact)

    print("\n=== Global extremes for C_ext_mean ===")
    best = df.sort_values("C_ext_mean", ascending=False).iloc[[0]]
    worst = df.sort_values("C_ext_mean", ascending=True).iloc[[0]]
    print("\nBest run by C_ext_mean:\n")
    if _HAS_IPY:
        display(best)
    else:
        print(best)
    print("\nWorst run by C_ext_mean:\n")
    if _HAS_IPY:
        display(worst)
    else:
        print(worst)

    print("\n=== Average mode dwell fractions by kappa_route ===\n")
    dwell_kappa = df.groupby("kappa_route")[["frac_supercoherent", "frac_nominal",
                                             "frac_drift", "frac_alert"]].mean()
    if _HAS_IPY:
        display(dwell_kappa)
    else:
        print(dwell_kappa)

    print("\n=== Average mode dwell fractions by whisper_coupling ===\n")
    dwell_wc = df.groupby("whisper_coupling")[["frac_supercoherent", "frac_nominal",
                                               "frac_drift", "frac_alert"]].mean()
    if _HAS_IPY:
        display(dwell_wc)
    else:
        print(dwell_wc)

    plot_benchmark_figures(df)
