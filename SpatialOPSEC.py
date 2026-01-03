"""
SPATIAL-OPSEC+INT · NG v3.1 (Single-File, Production-Oriented)

Next-generation spatial environmental intelligence testbed.
Focus: sensing, situational awareness, safety, and research – not weapon control.

Key capabilities:
- Multi-scenario environment-as-agent (open field, urban, forest, coastal, mountain, etc.)
- Multi-modal sensor suites (RGB, thermal, radar, acoustic) with self-calibrating trust
- Multi-channel semantic field (generic activity, mobile/moving, periodic/acoustic)
- Policy-driven dynamic deployment (coverage, hotspot, perimeter, hybrid)
- Domain catalog (OPSEC_OPEN_FIELD, URBAN_SECURITY, PERIMETER_MONITOR, CRITICAL_FACILITY,
  DISASTER_RESPONSE, FOREST_BORDER, COASTAL_PORT, MOUNTAIN_PASS, DESERT_TEST, ARCTIC_OUTPOST)
- Application catalog (MESH_NETWORKING, ROBOTICS_SWARM, SMART_CITY, LOGISTICS_FLOW,
  ENERGY_GRID, MEDICAL_NEURO, FINANCE_EXECUTION, HUMAN_SYSTEMS)
- Digital-twin hooks (what-if cloning, domain + application contexts)
- Benchmarks, policy sweeps, telemetry export with privacy masking
- Risk / error analysis (blindspots, overconfidence, underconfidence, cost-efficiency)
- Sensor-level analytics and visualizations
- Experiment manager for multi-domain, multi-policy, multi-seed sweeps
- Text report generator for technical and application summaries
- Policy optimizer for auto-selecting the best deployment policy
- Engine facade for clean programmatic use (services, CLIs, notebooks)

Intended for Colab / notebooks and script environments.
Requires numpy and (optionally) matplotlib.
"""

from __future__ import annotations

import math
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ============================================================
#  LOGGING (production-style)
# ============================================================

logger = logging.getLogger("spatial_opsec_ng")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class SpatialConfigNG:
    # Grid
    grid_size_x: int = 80
    grid_size_y: int = 80
    cell_size_m: float = 1.0

    # Time
    dt_s: float = 0.5
    sim_steps: int = 200

    # Scenario descriptor
    # "open_field" | "urban" | "perimeter" | "forest" | "coastal" | "industrial" | "mountain"
    scenario: str = "open_field"

    # Environment dynamics
    num_hotspots: int = 5
    hotspot_move_prob: float = 0.15
    hotspot_max_step: int = 2

    # Terrain / attenuation
    use_terrain: bool = True
    terrain_attenuation_buildings: float = 0.3

    # Sensors
    base_sensor_range_m: float = 25.0
    base_sensor_fov_deg: float = 120.0

    # Fusion / uncertainty
    prior_activity_prob: float = 0.05
    sensor_detection_boost: float = 0.60
    sensor_false_positive_boost: float = 0.10
    sensor_miss_decay: float = 0.95

    # Semantic channels (K=3 for now)
    # 0: generic activity, 1: mobile / moving, 2: periodic / acoustic
    semantic_channels: int = 3

    # Asset manager
    max_dynamic_pods: int = 16
    deploy_interval_steps: int = 15
    uncertainty_threshold: float = 0.25
    min_distance_between_pods_m: float = 5.0

    # Cost model
    cost_drone: float = 10.0
    cost_tower: float = 8.0
    cost_ground_pod: float = 2.0
    cost_human: float = 1.0
    cost_launched_pod: float = 1.5

    # Sensor self-calibration
    trust_init: float = 0.8
    trust_min: float = 0.1
    trust_max: float = 1.5
    trust_lr: float = 0.02  # learning rate per step
    calib_drift_std: float = 0.003

    # Benchmarks
    benchmark_runs: int = 4
    benchmark_steps: int = 200

    # Randomness
    base_seed: int = 17

    # Privacy / telemetry
    privacy_noise_sigma: float = 0.01
    privacy_mask_fraction: float = 0.05  # fraction of cells randomly masked

    # Verbosity
    verbose: bool = True

    # Production toggles
    enable_risk_analysis: bool = True
    enable_telemetry_export: bool = True

    # Optional label for experiment tagging
    tag: str = "default"


# ============================================================
#  DOMAIN PROFILES
# ============================================================

@dataclass
class DomainProfileNG:
    name: str
    scenario: str
    sensor_suite: str
    default_policy: str
    description: str
    subs: Dict[str, Dict[str, str]]


DOMAIN_CATALOG_NG: Dict[str, DomainProfileNG] = {
    "OPSEC_OPEN_FIELD": DomainProfileNG(
        name="OPSEC_OPEN_FIELD",
        scenario="open_field",
        sensor_suite="mixed",
        default_policy="coverage",
        description="Open-field awareness with mixed airborne and ground sensors.",
        subs={
            "global_coverage": {"policy": "coverage", "notes": "Uniform coverage and low uncertainty."},
            "hotspot_tracking": {"policy": "hotspot", "notes": "Strong-but-uncertain activity pockets."},
            "balanced": {"policy": "hybrid", "notes": "Balanced perimeter + hotspot behaviour."},
        },
    ),
    "URBAN_SECURITY": DomainProfileNG(
        name="URBAN_SECURITY",
        scenario="urban",
        sensor_suite="mixed",
        default_policy="hotspot",
        description="Urban grid with partial occlusion and clustered hotspots.",
        subs={
            "infrastructure_monitoring": {"policy": "coverage", "notes": "Even coverage over core urban block."},
            "crowd_monitoring": {"policy": "hotspot", "notes": "Evolving crowd-like hotspots."},
            "hotspot_focus": {"policy": "hotspot", "notes": "Aggressive hotspot deployment."},
        },
    ),
    "PERIMETER_MONITOR": DomainProfileNG(
        name="PERIMETER_MONITOR",
        scenario="perimeter",
        sensor_suite="sparse_long_range",
        default_policy="perimeter",
        description="Perimeter-heavy environment; long-range boundary watch.",
        subs={
            "ring_coverage": {"policy": "perimeter", "notes": "Circular boundary coverage."},
            "breach_focus": {"policy": "hybrid", "notes": "Perimeter plus inferred breach hotspots."},
        },
    ),
    "CRITICAL_FACILITY": DomainProfileNG(
        name="CRITICAL_FACILITY",
        scenario="industrial",
        sensor_suite="industrial_dense",
        default_policy="coverage",
        description="Dense industrial footprint with high sensor density.",
        subs={
            "dense_coverage": {"policy": "coverage", "notes": "Dense local certainty."},
            "fallback": {"policy": "hybrid", "notes": "Degraded/fallback sensing mix."},
        },
    ),
    "DISASTER_RESPONSE": DomainProfileNG(
        name="DISASTER_RESPONSE",
        scenario="urban",
        sensor_suite="mixed",
        default_policy="coverage",
        description="Urban disaster response; search, triage and situational awareness.",
        subs={
            "urban_search": {"policy": "hotspot", "notes": "Likely survivor pockets under occlusion."},
            "perimeter_triage": {"policy": "perimeter", "notes": "Perimeter picture around incident zone."},
        },
    ),
    "FOREST_BORDER": DomainProfileNG(
        name="FOREST_BORDER",
        scenario="forest",
        sensor_suite="forest_border",
        default_policy="coverage",
        description="Forest border awareness with ground nets and drones.",
        subs={
            "border_watch": {"policy": "perimeter", "notes": "Forest edge crossings."},
            "deep_patrol": {"policy": "hybrid", "notes": "Edge plus deeper patrol sensing."},
        },
    ),
    "COASTAL_PORT": DomainProfileNG(
        name="COASTAL_PORT",
        scenario="coastal",
        sensor_suite="coastal_port",
        default_policy="coverage",
        description="Coastal port baseline picture.",
        subs={
            "harbor_coverage": {"policy": "coverage", "notes": "Harbor basin and quay coverage."},
            "channel_focus": {"policy": "hotspot", "notes": "Entry/exit lane focus."},
        },
    ),
    "MOUNTAIN_PASS": DomainProfileNG(
        name="MOUNTAIN_PASS",
        scenario="mountain",
        sensor_suite="mountain_pass_lr",
        default_policy="perimeter",
        description="Mountain pass ridge and valley sensing.",
        subs={
            "pass_watch": {"policy": "perimeter", "notes": "Pass boundary watch."},
            "valley_focus": {"policy": "hybrid", "notes": "Ridge plus valley floor."},
        },
    ),
}


# ============================================================
#  ENVIRONMENT-AS-AGENT
# ============================================================

class EnvironmentNG:
    """
    Environment-as-agent.
    - activity_field[y,x] in [0,1]
    - semantic_map[y,x] ∈ {0,1,2} (background, mobile-ish, periodic-ish)
    - terrain[y,x] ∈ [0,1] visibility factor
    """

    def __init__(self, cfg: SpatialConfigNG, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.activity = np.zeros((cfg.grid_size_y, cfg.grid_size_x), dtype=np.float32)
        self.semantic_map = np.zeros_like(self.activity, dtype=np.int8)
        self.terrain = np.ones_like(self.activity, dtype=np.float32)
        self.hotspots: List[Tuple[int, int, int]] = []  # (x,y,semantic_type)
        self._init_hotspots()
        self._init_terrain()

    def _init_hotspots(self):
        self.hotspots.clear()
        s = self.cfg.scenario
        kinds = [1, 2]  # mobile, periodic

        def add_hotspot(x, y):
            kind = int(self.rng.choice(kinds))
            self.hotspots.append((x, y, kind))

        if s in ("open_field", "forest"):
            for _ in range(self.cfg.num_hotspots):
                x = self.rng.integers(0, self.cfg.grid_size_x)
                y = self.rng.integers(0, self.cfg.grid_size_y)
                add_hotspot(x, y)
        elif s in ("urban", "industrial"):
            cx = self.cfg.grid_size_x // 2
            cy = self.cfg.grid_size_y // 2
            for _ in range(self.cfg.num_hotspots):
                dx = int(self.rng.normal(0, self.cfg.grid_size_x * 0.15))
                dy = int(self.rng.normal(0, self.cfg.grid_size_y * 0.15))
                x = int(np.clip(cx + dx, 0, self.cfg.grid_size_x - 1))
                y = int(np.clip(cy + dy, 0, self.cfg.grid_size_y - 1))
                add_hotspot(x, y)
        else:  # perimeter / coastal / mountain etc.
            for _ in range(self.cfg.num_hotspots):
                side = self.rng.integers(0, 4)
                if side == 0:
                    x = self.rng.integers(0, self.cfg.grid_size_x)
                    y = 0
                elif side == 1:
                    x = self.rng.integers(0, self.cfg.grid_size_x)
                    y = self.cfg.grid_size_y - 1
                elif side == 2:
                    x = 0
                    y = self.rng.integers(0, self.cfg.grid_size_y)
                else:
                    x = self.cfg.grid_size_x - 1
                    y = self.rng.integers(0, self.cfg.grid_size_y)
                add_hotspot(x, y)

        self._rebuild_fields()

    def _init_terrain(self):
        self.terrain.fill(1.0)
        if not self.cfg.use_terrain:
            return
        s = self.cfg.scenario
        h, w = self.terrain.shape

        if s in ("urban", "industrial"):
            block = max(6, min(h, w) // 8)
            for by in range(2, h, block * 2):
                for bx in range(2, w, block * 2):
                    y0 = by
                    y1 = min(by + block, h)
                    x0 = bx
                    x1 = min(bx + block, w)
                    self.terrain[y0:y1, x0:x1] = self.cfg.terrain_attenuation_buildings
        elif s == "perimeter":
            margin = min(h, w) // 4
            self.terrain[margin:-margin, margin:-margin] *= 0.8
        elif s == "forest":
            num_patches = max(10, (h * w) // 200)
            for _ in range(num_patches):
                cx = self.rng.integers(0, w)
                cy = self.rng.integers(0, h)
                rx = self.rng.integers(3, 8)
                ry = self.rng.integers(3, 8)
                x0 = max(0, cx - rx)
                x1 = min(w, cx + rx)
                y0 = max(0, cy - ry)
                y1 = min(h, cy + ry)
                self.terrain[y0:y1, x0:x1] *= 0.4
            for _ in range(max(4, num_patches // 4)):
                cx = self.rng.integers(0, w)
                cy = self.rng.integers(0, h)
                rx = self.rng.integers(4, 10)
                ry = self.rng.integers(4, 10)
                x0 = max(0, cx - rx)
                x1 = min(w, cx + rx)
                y0 = max(0, cy - ry)
                y1 = min(h, cy + ry)
                self.terrain[y0:y1, x0:x1] = 1.0
        elif s == "coastal":
            water_rows = h // 3
            self.terrain[h - water_rows :, :] = 1.0
            quay_rows = max(3, h // 10)
            self.terrain[h - water_rows - quay_rows : h - water_rows, :] *= 0.8
            block = max(6, w // 10)
            for by in range(2, h - water_rows - quay_rows, block * 2):
                for bx in range(2, w, block * 2):
                    y0 = by
                    y1 = min(by + block, h - water_rows - quay_rows)
                    x0 = bx
                    x1 = min(bx + block, w)
                    self.terrain[y0:y1, x0:x1] = self.cfg.terrain_attenuation_buildings
        elif s == "mountain":
            for y in range(h):
                for x in range(w):
                    d = abs(y - (h - 1 - x))
                    if d < 2:
                        self.terrain[y, x] *= 0.2
                    elif d < 4:
                        self.terrain[y, x] *= 0.5

    def _rebuild_fields(self):
        self.activity.fill(0.0)
        self.semantic_map.fill(0)

        for (hx, hy, kind) in self.hotspots:
            for y in range(self.cfg.grid_size_y):
                for x in range(self.cfg.grid_size_x):
                    dx = x - hx
                    dy = y - hy
                    d2 = dx * dx + dy * dy
                    if d2 == 0:
                        contrib = 1.0
                    else:
                        contrib = 1.0 / (1.0 + 0.1 * d2)
                    self.activity[y, x] += contrib
                    if contrib > 0.2:
                        if kind == 1:
                            self.semantic_map[y, x] = 1
                        elif kind == 2:
                            self.semantic_map[y, x] = 2

        max_val = float(self.activity.max()) if self.activity.max() > 0 else 1.0
        self.activity /= max_val

        s = self.cfg.scenario
        if s in ("urban", "industrial"):
            for y in range(0, self.cfg.grid_size_y, 6):
                self.activity[y, :] *= 0.7
            for x in range(0, self.cfg.grid_size_x, 6):
                self.activity[:, x] *= 0.7
        if s in ("perimeter", "coastal", "mountain"):
            margin = 4
            self.activity[:margin, :] *= 1.2
            self.activity[-margin:, :] *= 1.2
            self.activity[:, :margin] *= 1.2
            self.activity[:, -margin:] *= 1.2
            self.activity = np.clip(self.activity, 0.0, 1.0)

    def step(self):
        moved = False
        new_hotspots: List[Tuple[int, int, int]] = []
        s = self.cfg.scenario

        for (hx, hy, kind) in self.hotspots:
            move_prob = self.cfg.hotspot_move_prob
            max_step = self.cfg.hotspot_max_step
            if s in ("urban", "industrial"):
                move_prob *= 0.75
                max_step = max(1, max_step - 1)
            elif s in ("perimeter", "coastal", "mountain"):
                if hx in (0, self.cfg.grid_size_x - 1) or hy in (0, self.cfg.grid_size_y - 1):
                    move_prob *= 0.6
            if self.rng.random() < move_prob:
                dx = self.rng.integers(-max_step, max_step + 1)
                dy = self.rng.integers(-max_step, max_step + 1)
                nx = int(np.clip(hx + dx, 0, self.cfg.grid_size_x - 1))
                ny = int(np.clip(hy + dy, 0, self.cfg.grid_size_y - 1))
                new_hotspots.append((nx, ny, kind))
                moved = True
            else:
                new_hotspots.append((hx, hy, kind))

        self.hotspots = new_hotspots
        if moved:
            self._rebuild_fields()


# ============================================================
#  SENSORS WITH SELF-CALIBRATION
# ============================================================

class Modality:
    RGB = "rgb_camera"
    THERMAL = "thermal"
    RADAR = "radar"
    ACOUSTIC = "acoustic"


@dataclass
class SensorNodeNG:
    node_id: str
    kind: str
    modality: str
    x: float
    y: float
    z: float
    fov_deg: float
    range_m: float
    noise_std: float
    update_rate_hz: float
    tethered: bool = False
    mobile: bool = False
    active: bool = True
    metadata: Dict[str, float] = field(default_factory=dict)
    last_update_t: float = 0.0
    trust: float = 1.0
    calib_bias: float = 0.0
    last_obs: Optional[np.ndarray] = None

    def can_update(self, t: float) -> bool:
        min_dt = 1.0 / max(self.update_rate_hz, 1e-6)
        return self.active and (t - self.last_update_t) >= min_dt

    def world_to_grid(self, cfg: SpatialConfigNG) -> Tuple[int, int]:
        gx = int(np.clip(self.x / cfg.cell_size_m, 0, cfg.grid_size_x - 1))
        gy = int(np.clip(self.y / cfg.cell_size_m, 0, cfg.grid_size_y - 1))
        return gx, gy

    def _modality_gain_and_channel(self) -> Tuple[float, int]:
        """
        Returns (gain, semantic_channel_index):
        0: generic, 1: mobile-ish, 2: periodic-ish
        """
        if self.modality == Modality.THERMAL:
            return 1.4, 0
        if self.modality == Modality.RGB:
            return 1.0, 0
        if self.modality == Modality.RADAR:
            return 1.2, 1
        if self.modality == Modality.ACOUSTIC:
            return 0.8, 2
        return 1.0, 0

    def drift_calibration(self, cfg: SpatialConfigNG, rng: np.random.Generator):
        self.calib_bias += rng.normal(0.0, cfg.calib_drift_std)

    def sense(
        self,
        cfg: SpatialConfigNG,
        env: EnvironmentNG,
        t: float,
        rng: np.random.Generator,
    ) -> Optional[Tuple[np.ndarray, int]]:
        if not self.can_update(t):
            return None
        self.last_update_t = t
        self.drift_calibration(cfg, rng)

        gx, gy = self.world_to_grid(cfg)
        r_cells = int(self.range_m / cfg.cell_size_m)
        max_r2 = r_cells * r_cells

        obs = np.zeros_like(env.activity)
        gain, semantic_idx = self._modality_gain_and_channel()
        fov_rad = math.radians(self.fov_deg)
        fov_factor = fov_rad / math.pi if self.fov_deg > 0 else 0.0

        env_signal = (2.0 * env.activity - 1.0)

        for y in range(env.cfg.grid_size_y):
            for x in range(env.cfg.grid_size_x):
                dx = x - gx
                dy = y - gy
                d2 = dx * dx + dy * dy
                if d2 > max_r2:
                    continue
                d = math.sqrt(d2) + 1e-6
                range_factor = max(0.0, 1.0 - d / (r_cells + 1e-6))
                terrain_factor = env.terrain[y, x] if cfg.use_terrain else 1.0
                signal = env_signal[y, x] * gain * range_factor * fov_factor * terrain_factor
                noise = rng.normal(loc=self.calib_bias, scale=self.noise_std)
                obs[y, x] = self.trust * (signal + noise)

        self.last_obs = obs
        return obs, semantic_idx


# ============================================================
#  FUSION ENGINE WITH SEMANTICS
# ============================================================

class FusionNG:
    """
    Maintains:
    - activity_prob[y,x]
    - semantic_probs[k,y,x] over K channels (K = cfg.semantic_channels)
    """

    def __init__(self, cfg: SpatialConfigNG):
        self.cfg = cfg
        self.activity_prob = np.full(
            (cfg.grid_size_y, cfg.grid_size_x),
            cfg.prior_activity_prob,
            dtype=np.float32,
        )
        self.semantic_probs = np.full(
            (cfg.semantic_channels, cfg.grid_size_y, cfg.grid_size_x),
            1.0 / cfg.semantic_channels,
            dtype=np.float32,
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fuse_observation(
        self,
        obs: np.ndarray,
        semantic_idx: int,
    ):
        cfg = self.cfg
        evidence = self._sigmoid(obs * 2.5)
        eps = 1e-5
        evidence = np.clip(evidence, eps, 1.0 - eps)

        p = self.activity_prob
        p_seen_active = np.clip(
            p + cfg.sensor_detection_boost * (evidence - 0.5),
            0.0,
            1.0,
        )
        p_seen_inactive = np.clip(
            p - cfg.sensor_false_positive_boost * (0.5 - evidence),
            0.0,
            1.0,
        )
        alpha = np.clip(np.abs(obs), 0.0, 1.0)
        new_p = alpha * p_seen_active + (1.0 - alpha) * p_seen_inactive
        self.activity_prob = new_p.astype(np.float32)

        # Semantic update — push semantic_probs[semantic_idx] towards evidence
        K = self.semantic_probs.shape[0]
        sem = self.semantic_probs
        w = np.clip(np.abs(obs), 0.0, 1.0)
        w = w.astype(np.float32)

        e_sem = evidence

        sem_target = sem[semantic_idx]
        sem[semantic_idx] = (1 - w) * sem_target + w * e_sem

        for k in range(K):
            if k == semantic_idx:
                continue
            sem[k] = (1 - 0.25 * w) * sem[k]

        sum_sem = sem.sum(axis=0, keepdims=True) + 1e-9
        self.semantic_probs = (sem / sum_sem).astype(np.float32)

    def decay_unseen(self, mask_observed: np.ndarray):
        self.activity_prob[mask_observed] *= self.cfg.sensor_miss_decay
        unseen = ~mask_observed
        K = self.semantic_probs.shape[0]
        uniform = 1.0 / K
        self.semantic_probs[:, unseen] = (
            0.9 * self.semantic_probs[:, unseen] + 0.1 * uniform
        )

    def coverage_fraction(self, threshold: float = 0.1) -> float:
        return float((self.activity_prob > threshold).mean())

    def average_confidence(self) -> float:
        return float(np.abs(self.activity_prob - 0.5).mean()) * 2.0

    def mean_entropy_bits(self) -> float:
        p = self.activity_prob
        eps = 1e-6
        h = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
        h_bits = h / math.log(2.0)
        return float(h_bits.mean())

    def semantic_mode_map(self) -> np.ndarray:
        return np.argmax(self.semantic_probs, axis=0).astype(np.int8)

    def uncertainty_mask(self, margin: float) -> np.ndarray:
        p = self.activity_prob
        return (p > (0.5 - margin)) & (p < (0.5 + margin))


# ============================================================
#  DEPLOYMENT POLICIES
# ============================================================

class BasePolicyNG:
    def __init__(self, cfg: SpatialConfigNG, name: str):
        self.cfg = cfg
        self.name = name

    def select_cells(
        self,
        fusion: FusionNG,
        env: EnvironmentNG,
        sensors: List[SensorNodeNG],
        rng: np.random.Generator,
        budget: int,
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError


class CoveragePolicyNG(BasePolicyNG):
    def __init__(self, cfg: SpatialConfigNG):
        super().__init__(cfg, "coverage")

    def select_cells(self, fusion, env, sensors, rng, budget):
        mask = fusion.uncertainty_mask(self.cfg.uncertainty_threshold)
        ys, xs = np.where(mask)
        cells = list(zip(xs, ys))
        if not cells:
            return []
        rng.shuffle(cells)
        return cells[:budget]


class PerimeterPolicyNG(BasePolicyNG):
    def __init__(self, cfg: SpatialConfigNG):
        super().__init__(cfg, "perimeter")

    def select_cells(self, fusion, env, sensors, rng, budget):
        mask = fusion.uncertainty_mask(self.cfg.uncertainty_threshold)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        margin = max(3, min(self.cfg.grid_size_x, self.cfg.grid_size_y) // 10)
        candidates = []
        for x, y in zip(xs, ys):
            if (
                x < margin
                or x >= self.cfg.grid_size_x - margin
                or y < margin
                or y >= self.cfg.grid_size_y - margin
            ):
                candidates.append((x, y))
        if not candidates:
            return CoveragePolicyNG(self.cfg).select_cells(
                fusion, env, sensors, rng, budget
            )
        rng.shuffle(candidates)
        return candidates[:budget]


class HotspotPolicyNG(BasePolicyNG):
    def __init__(self, cfg: SpatialConfigNG):
        super().__init__(cfg, "hotspot")

    def select_cells(self, fusion, env, sensors, rng, budget):
        mask = fusion.uncertainty_mask(self.cfg.uncertainty_threshold)
        p = fusion.activity_prob
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        scores = []
        for x, y in zip(xs, ys):
            scores.append((p[y, x], x, y))
        scores.sort(reverse=True, key=lambda t: t[0])
        top = [(x, y) for (_, x, y) in scores[: budget * 4]]
        rng.shuffle(top)
        return top[:budget]


class HybridPolicyNG(BasePolicyNG):
    def __init__(self, cfg: SpatialConfigNG):
        super().__init__(cfg, "hybrid")
        self.cov = CoveragePolicyNG(cfg)
        self.per = PerimeterPolicyNG(cfg)
        self.hot = HotspotPolicyNG(cfg)

    def select_cells(self, fusion, env, sensors, rng, budget):
        half = max(1, budget // 2)
        hs = self.hot.select_cells(fusion, env, sensors, rng, half)
        per = self.per.select_cells(fusion, env, sensors, rng, budget - len(hs))
        combined = hs + per
        seen = set()
        out = []
        for c in combined:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out[:budget]


def make_policy_ng(name: str, cfg: SpatialConfigNG) -> BasePolicyNG:
    if name == "coverage":
        return CoveragePolicyNG(cfg)
    if name == "perimeter":
        return PerimeterPolicyNG(cfg)
    if name == "hotspot":
        return HotspotPolicyNG(cfg)
    if name == "hybrid":
        return HybridPolicyNG(cfg)
    return CoveragePolicyNG(cfg)


# ============================================================
#  ASSET MANAGER
# ============================================================

@dataclass
class AssetManagerNG:
    cfg: SpatialConfigNG
    sensors: List[SensorNodeNG]
    policy: BasePolicyNG
    dynamic_ids: List[str] = field(default_factory=list)
    attempts: int = 0
    successes: int = 0

    def _dynamic_positions(self) -> List[Tuple[float, float]]:
        out = []
        for s in self.sensors:
            if s.node_id in self.dynamic_ids and s.active:
                out.append((s.x, s.y))
        return out

    def _too_close(self, x: float, y: float) -> bool:
        min_d2 = self.cfg.min_distance_between_pods_m ** 2
        for px, py in self._dynamic_positions():
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < min_d2:
                return True
        return False

    def _deploy(self, gx: int, gy: int):
        x = (gx + 0.5) * self.cfg.cell_size_m
        y = (gy + 0.5) * self.cfg.cell_size_m
        if self._too_close(x, y):
            return False
        nid = f"dyn_pod_{len(self.dynamic_ids)}"
        pod = SensorNodeNG(
            node_id=nid,
            kind="launched_pod",
            modality=Modality.THERMAL,
            x=x,
            y=y,
            z=0.5,
            fov_deg=360.0,
            range_m=self.cfg.base_sensor_range_m * 0.9,
            noise_std=0.13,
            update_rate_hz=1.0,
            tethered=False,
            mobile=False,
            active=True,
            metadata={"role": f"dyn_{self.policy.name}"},
            trust=self.cfg.trust_init,
        )
        self.sensors.append(pod)
        self.dynamic_ids.append(nid)
        return True

    def maybe_deploy(
        self,
        step_idx: int,
        fusion: FusionNG,
        env: EnvironmentNG,
        rng: np.random.Generator,
    ):
        if step_idx % self.cfg.deploy_interval_steps != 0:
            return
        if len(self.dynamic_ids) >= self.cfg.max_dynamic_pods:
            return
        self.attempts += 1
        budget = max(1, self.cfg.max_dynamic_pods // 4)
        cells = self.policy.select_cells(fusion, env, self.sensors, rng, budget)
        deployed = 0
        for gx, gy in cells:
            if len(self.dynamic_ids) >= self.cfg.max_dynamic_pods:
                break
            if self._deploy(gx, gy):
                deployed += 1
        if deployed > 0:
            self.successes += deployed
            if self.cfg.verbose:
                logger.info(
                    f"[ASSET] step={step_idx} policy={self.policy.name} "
                    f"deployed {deployed} launched pods (total={len(self.dynamic_ids)})"
                )


# ============================================================
#  ORCHESTRATOR WITH SENSOR TRUST UPDATE
# ============================================================

class OrchestratorNG:
    def __init__(
        self,
        cfg: SpatialConfigNG,
        sensors: List[SensorNodeNG],
        policy: BasePolicyNG,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None else cfg.base_seed)
        self.env = EnvironmentNG(cfg, self.rng)
        self.fusion = FusionNG(cfg)
        self.sensors = sensors
        for s in self.sensors:
            if not hasattr(s, "trust") or s.trust is None:
                s.trust = cfg.trust_init
        self.assets = AssetManagerNG(cfg, self.sensors, policy)
        self.t = 0.0
        self.step_idx = 0

        self.static_cost = self._compute_static_cost()
        self.metrics: Dict[str, List[float]] = {
            "coverage": [],
            "confidence": [],
            "env_correlation": [],
            "entropy_bits": [],
            "dynamic_pods": [],
            "dynamic_cost": [],
        }

    def _compute_static_cost(self) -> float:
        c = 0.0
        for s in self.sensors:
            if s.kind in ("drone", "patrol_drone", "balloon", "satellite"):
                c += self.cfg.cost_drone
            elif s.kind in ("tower", "harbor_tower", "ridge_tower"):
                c += self.cfg.cost_tower
            elif s.kind in ("ground_pod", "forest_pod", "valley_pod"):
                c += self.cfg.cost_ground_pod
            elif s.kind == "human":
                c += self.cfg.cost_human
            elif s.kind == "launched_pod":
                continue
            else:
                c += self.cfg.cost_ground_pod
        return c

    def _dynamic_cost(self) -> float:
        return len(self.assets.dynamic_ids) * self.cfg.cost_launched_pod

    def _env_correlation(self) -> float:
        gt = self.env.activity.flatten()
        est = self.fusion.activity_prob.flatten()
        gt_mean = float(gt.mean())
        est_mean = float(est.mean())
        num = float(((gt - gt_mean) * (est - est_mean)).mean())
        denom = float(gt.std() * est.std() + 1e-8)
        return num / denom if denom > 0 else 0.0

    def _update_sensor_trusts(self):
        cfg = self.cfg
        env_signal = (2.0 * self.env.activity - 1.0)
        for s in self.sensors:
            if s.last_obs is None:
                continue
            obs = s.last_obs
            corr = float((obs * env_signal).mean())
            delta = cfg.trust_lr if corr > 0 else -cfg.trust_lr
            s.trust = float(np.clip(s.trust + delta, cfg.trust_min, cfg.trust_max))

    def step(self):
        cfg = self.cfg
        self.env.step()

        combined_mask = np.zeros_like(self.env.activity, dtype=bool)

        for s in self.sensors:
            res = s.sense(cfg, self.env, self.t, self.rng)
            if res is None:
                continue
            obs, sem_idx = res
            self.fusion.fuse_observation(obs, sem_idx)
            combined_mask |= (np.abs(obs) > 1e-3)

        if combined_mask.any():
            self.fusion.decay_unseen(combined_mask)

        self._update_sensor_trusts()
        self.assets.maybe_deploy(self.step_idx, self.fusion, self.env, self.rng)

        cov = self.fusion.coverage_fraction()
        conf = self.fusion.average_confidence()
        corr = self._env_correlation()
        Hbits = self.fusion.mean_entropy_bits()
        dyn_pods = len(self.assets.dynamic_ids)
        dyn_cost = self._dynamic_cost()

        self.metrics["coverage"].append(cov)
        self.metrics["confidence"].append(conf)
        self.metrics["env_correlation"].append(corr)
        self.metrics["entropy_bits"].append(Hbits)
        self.metrics["dynamic_pods"].append(dyn_pods)
        self.metrics["dynamic_cost"].append(dyn_cost)

        if cfg.verbose and (self.step_idx % 25 == 0):
            logger.info(
                f"[STEP {self.step_idx:03d}] "
                f"cov={cov:.3f} conf={conf:.3f} env_corr={corr:.3f} "
                f"H={Hbits:.3f} dyn_pods={dyn_pods} dyn_cost={dyn_cost:.1f}"
            )

        self.step_idx += 1        # increment step index
        self.t += cfg.dt_s

    def run(self, steps: Optional[int] = None):
        steps = steps if steps is not None else self.cfg.sim_steps
        for _ in range(steps):
            self.step()

    def summarize(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, arr in self.metrics.items():
            if not arr:
                out[f"{k}_mean"] = 0.0
                out[f"{k}_final"] = 0.0
            else:
                v = np.array(arr, dtype=np.float32)
                out[f"{k}_mean"] = float(v.mean())
                out[f"{k}_final"] = float(v[-1])
        out["deploy_attempts"] = float(self.assets.attempts)
        out["deploy_success"] = float(self.assets.successes)
        out["static_asset_cost"] = float(self.static_cost)
        out["dynamic_cost_final"] = float(self.metrics["dynamic_cost"][-1]) if self.metrics["dynamic_cost"] else 0.0
        total = out["static_asset_cost"] + out["dynamic_cost_final"]
        out["total_asset_cost_final"] = total
        dyn_cost = out["dynamic_cost_final"]
        if dyn_cost > 0:
            out["env_corr_per_dyn_cost"] = out["env_correlation_final"] / dyn_cost
        else:
            out["env_corr_per_dyn_cost"] = 0.0
        return out


# ============================================================
#  SENSOR SUITES
# ============================================================

def suite_mixed(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []

    radius = cfg.grid_size_x * cfg.cell_size_m * 0.4
    num_drones = 6
    for i in range(num_drones):
        angle = 2 * math.pi * i / num_drones
        x = cfg.grid_size_x * cfg.cell_size_m / 2 + radius * math.cos(angle)
        y = cfg.grid_size_y * cfg.cell_size_m / 2 + radius * math.sin(angle)
        modality = Modality.RGB if i % 2 == 0 else Modality.THERMAL
        sensors.append(
            SensorNodeNG(
                node_id=f"drone_{i}",
                kind="drone",
                modality=modality,
                x=x,
                y=y,
                z=60.0,
                fov_deg=cfg.base_sensor_fov_deg,
                range_m=cfg.base_sensor_range_m * 1.6,
                noise_std=0.08,
                update_rate_hz=2.0,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "overwatch"},
                trust=cfg.trust_init,
            )
        )

    corners = [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]
    for i, (fx, fy) in enumerate(corners):
        x = fx * cfg.grid_size_x * cfg.cell_size_m
        y = fy * cfg.grid_size_y * cfg.cell_size_m
        sensors.append(
            SensorNodeNG(
                node_id=f"tower_{i}",
                kind="tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=70.0,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 2.2,
                noise_std=0.10,
                update_rate_hz=0.7,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "boundary_watch"},
                trust=cfg.trust_init,
            )
        )

    num_pods = 16
    for i in range(num_pods):
        x = (i + 0.5) * cfg.grid_size_x * cfg.cell_size_m / num_pods
        y = cfg.grid_size_y * cfg.cell_size_m * 0.35
        modality = Modality.THERMAL if i % 2 == 0 else Modality.ACOUSTIC
        sensors.append(
            SensorNodeNG(
                node_id=f"pod_{i}",
                kind="ground_pod",
                modality=modality,
                x=x,
                y=y,
                z=0.5,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 0.8,
                noise_std=0.15,
                update_rate_hz=1.0,
                tethered=False,
                mobile=False,
                active=True,
                metadata={"role": "grid"},
                trust=cfg.trust_init,
            )
        )

    sensors.append(
        SensorNodeNG(
            node_id="human_0",
            kind="human",
            modality=Modality.RGB,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.1,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.9,
            z=1.7,
            fov_deg=90.0,
            range_m=cfg.base_sensor_range_m,
            noise_std=0.18,
            update_rate_hz=0.7,
            tethered=False,
            mobile=True,
            active=True,
            metadata={"role": "patrol"},
            trust=cfg.trust_init,
        )
    )

    sensors.append(
        SensorNodeNG(
            node_id="launcher_0",
            kind="launcher",
            modality=Modality.ACOUSTIC,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.5,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.9,
            z=1.0,
            fov_deg=0.0,
            range_m=0.0,
            noise_std=1.0,
            update_rate_hz=0.1,
            tethered=False,
            mobile=True,
            active=False,
            metadata={"role": "deployment_origin"},
            trust=cfg.trust_init,
        )
    )
    return sensors


def suite_dense_ground(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []
    rows, cols = 5, 10
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * cfg.grid_size_x * cfg.cell_size_m / cols
            y = (r + 0.5) * cfg.grid_size_y * cfg.cell_size_m / rows
            modality = Modality.THERMAL if (r + c) % 2 == 0 else Modality.ACOUSTIC
            sensors.append(
                SensorNodeNG(
                    node_id=f"grid_pod_{r}_{c}",
                    kind="ground_pod",
                    modality=modality,
                    x=x,
                    y=y,
                    z=0.5,
                    fov_deg=360.0,
                    range_m=cfg.base_sensor_range_m * 0.7,
                    noise_std=0.18,
                    update_rate_hz=0.9,
                    tethered=False,
                    mobile=False,
                    active=True,
                    metadata={"role": "dense_grid"},
                    trust=cfg.trust_init,
                )
            )
    return sensors


def suite_sparse_long_range(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []
    towers = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    for i, (fx, fy) in enumerate(towers):
        x = fx * cfg.grid_size_x * cfg.cell_size_m
        y = fy * cfg.grid_size_y * cfg.cell_size_m
        sensors.append(
            SensorNodeNG(
                node_id=f"tower_lr_{i}",
                kind="tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=80.0,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 3.0,
                noise_std=0.11,
                update_rate_hz=0.6,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "long_range"},
                trust=cfg.trust_init,
            )
        )
    sensors.append(
        SensorNodeNG(
            node_id="drone_center",
            kind="drone",
            modality=Modality.THERMAL,
            x=cfg.grid_size_x * cfg.cell_size_m / 2,
            y=cfg.grid_size_y * cfg.cell_size_m / 2,
            z=50.0,
            fov_deg=180.0,
            range_m=cfg.base_sensor_range_m * 1.5,
            noise_std=0.09,
            update_rate_hz=1.5,
            tethered=False,
            mobile=True,
            active=True,
            metadata={"role": "thermal_hub"},
            trust=cfg.trust_init,
        )
    )
    return sensors


def suite_forest_border(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []
    num_towers = 5
    for i in range(num_towers):
        x = (i + 0.5) * cfg.grid_size_x * cfg.cell_size_m / num_towers
        y = cfg.grid_size_y * cfg.cell_size_m * 0.05
        sensors.append(
            SensorNodeNG(
                node_id=f"ridge_tower_{i}",
                kind="ridge_tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=50.0,
                fov_deg=260.0,
                range_m=cfg.base_sensor_range_m * 2.0,
                noise_std=0.11,
                update_rate_hz=0.7,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "forest_edge"},
                trust=cfg.trust_init,
            )
        )
    num_pods = 24
    for i in range(num_pods):
        x = cfg.grid_size_x * cfg.cell_size_m * np.clip(np.random.uniform(0.0, 1.0), 0.0, 1.0)
        y = cfg.grid_size_y * cfg.cell_size_m * np.random.uniform(0.1, 0.7)
        modality = Modality.ACOUSTIC if i % 3 else Modality.THERMAL
        sensors.append(
            SensorNodeNG(
                node_id=f"forest_pod_{i}",
                kind="forest_pod",
                modality=modality,
                x=x,
                y=y,
                z=0.5,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 0.6,
                noise_std=0.20,
                update_rate_hz=0.8,
                tethered=False,
                mobile=False,
                active=True,
                metadata={"role": "forest_net"},
                trust=cfg.trust_init,
            )
        )
    for i in range(3):
        x = cfg.grid_size_x * cfg.cell_size_m * (0.25 + 0.25 * i)
        y = cfg.grid_size_y * cfg.cell_size_m * 0.6
        sensors.append(
            SensorNodeNG(
                node_id=f"patrol_drone_{i}",
                kind="patrol_drone",
                modality=Modality.THERMAL,
                x=x,
                y=y,
                z=70.0,
                fov_deg=140.0,
                range_m=cfg.base_sensor_range_m * 1.4,
                noise_std=0.10,
                update_rate_hz=1.5,
                tethered=False,
                mobile=True,
                active=True,
                metadata={"role": "forest_overwatch"},
                trust=cfg.trust_init,
            )
        )
    sensors.append(
        SensorNodeNG(
            node_id="launcher_forest",
            kind="launcher",
            modality=Modality.ACOUSTIC,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.5,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.9,
            z=1.0,
            fov_deg=0.0,
            range_m=0.0,
            noise_std=1.0,
            update_rate_hz=0.1,
            tethered=False,
            mobile=True,
            active=False,
            metadata={"role": "forest_launcher"},
            trust=cfg.trust_init,
        )
    )
    return sensors


def suite_coastal_port(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []
    for i in range(4):
        x = (0.2 + 0.2 * i) * cfg.grid_size_x * cfg.cell_size_m
        y = cfg.grid_size_y * cfg.cell_size_m * 0.4
        sensors.append(
            SensorNodeNG(
                node_id=f"harbor_tower_{i}",
                kind="harbor_tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=60.0,
                fov_deg=320.0,
                range_m=cfg.base_sensor_range_m * 2.5,
                noise_std=0.10,
                update_rate_hz=0.8,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "harbor_surv"},
                trust=cfg.trust_init,
            )
        )
    num_quay = 18
    for i in range(num_quay):
        x = (i + 0.5) * cfg.grid_size_x * cfg.cell_size_m / num_quay
        y = cfg.grid_size_y * cfg.cell_size_m * 0.65
        modality = Modality.ACOUSTIC if i % 2 else Modality.THERMAL
        sensors.append(
            SensorNodeNG(
                node_id=f"quay_pod_{i}",
                kind="ground_pod",
                modality=modality,
                x=x,
                y=y,
                z=0.5,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 0.7,
                noise_std=0.18,
                update_rate_hz=0.9,
                tethered=False,
                mobile=False,
                active=True,
                metadata={"role": "quay_line"},
                trust=cfg.trust_init,
            )
        )
    for i in range(3):
        x = cfg.grid_size_x * cfg.cell_size_m * (0.3 + 0.2 * i)
        y = cfg.grid_size_y * cfg.cell_size_m * 0.15
        sensors.append(
            SensorNodeNG(
                node_id=f"harbor_drone_{i}",
                kind="patrol_drone",
                modality=Modality.RGB if i % 2 else Modality.THERMAL,
                x=x,
                y=y,
                z=80.0,
                fov_deg=150.0,
                range_m=cfg.base_sensor_range_m * 1.8,
                noise_std=0.09,
                update_rate_hz=1.7,
                tethered=False,
                mobile=True,
                active=True,
                metadata={"role": "channel_patrol"},
                trust=cfg.trust_init,
            )
        )
    sensors.append(
        SensorNodeNG(
            node_id="launcher_coastal",
            kind="launcher",
            modality=Modality.ACOUSTIC,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.1,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.9,
            z=1.0,
            fov_deg=0.0,
            range_m=0.0,
            noise_std=1.0,
            update_rate_hz=0.1,
            tethered=False,
            mobile=True,
            active=False,
            metadata={"role": "coastal_launcher"},
            trust=cfg.trust_init,
        )
    )
    return sensors


def suite_industrial_dense(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors = suite_dense_ground(cfg)
    for i, fx in enumerate([0.1, 0.9]):
        x = fx * cfg.grid_size_x * cfg.cell_size_m
        y = cfg.grid_size_y * cfg.cell_size_m * 0.5
        sensors.append(
            SensorNodeNG(
                node_id=f"ind_tower_{i}",
                kind="tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=70.0,
                fov_deg=260.0,
                range_m=cfg.base_sensor_range_m * 2.0,
                noise_std=0.10,
                update_rate_hz=0.7,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "industrial_boundary"},
                trust=cfg.trust_init,
            )
        )
    for i in range(2):
        x = cfg.grid_size_x * cfg.cell_size_m * (0.3 + 0.4 * i)
        y = cfg.grid_size_y * cfg.cell_size_m * 0.6
        sensors.append(
            SensorNodeNG(
                node_id=f"ind_survey_{i}",
                kind="drone",
                modality=Modality.THERMAL,
                x=x,
                y=y,
                z=80.0,
                fov_deg=150.0,
                range_m=cfg.base_sensor_range_m * 1.7,
                noise_std=0.09,
                update_rate_hz=1.3,
                tethered=False,
                mobile=True,
                active=True,
                metadata={"role": "industrial_overwatch"},
                trust=cfg.trust_init,
            )
        )
    sensors.append(
        SensorNodeNG(
            node_id="launcher_industrial",
            kind="launcher",
            modality=Modality.ACOUSTIC,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.5,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.95,
            z=1.0,
            fov_deg=0.0,
            range_m=0.0,
            noise_std=1.0,
            update_rate_hz=0.1,
            tethered=False,
            mobile=True,
            active=False,
            metadata={"role": "industrial_launcher"},
            trust=cfg.trust_init,
        )
    )
    return sensors


def suite_mountain_pass(cfg: SpatialConfigNG) -> List[SensorNodeNG]:
    sensors: List[SensorNodeNG] = []
    for i in range(5):
        t = (i + 0.5) / 5.0
        x = t * cfg.grid_size_x * cfg.cell_size_m
        y = (1.0 - t) * cfg.grid_size_y * cfg.cell_size_m * 0.3
        sensors.append(
            SensorNodeNG(
                node_id=f"ridge_tower_mp_{i}",
                kind="ridge_tower",
                modality=Modality.RADAR,
                x=x,
                y=y,
                z=90.0,
                fov_deg=260.0,
                range_m=cfg.base_sensor_range_m * 2.8,
                noise_std=0.11,
                update_rate_hz=0.6,
                tethered=True,
                mobile=False,
                active=True,
                metadata={"role": "ridge_watch"},
                trust=cfg.trust_init,
            )
        )
    num_valley = 20
    for i in range(num_valley):
        x = cfg.grid_size_x * cfg.cell_size_m * np.random.uniform(0.15, 0.85)
        y = cfg.grid_size_y * cfg.cell_size_m * np.random.uniform(0.4, 0.8)
        modality = Modality.ACOUSTIC if i % 2 else Modality.THERMAL
        sensors.append(
            SensorNodeNG(
                node_id=f"valley_pod_{i}",
                kind="valley_pod",
                modality=modality,
                x=x,
                y=y,
                z=0.5,
                fov_deg=360.0,
                range_m=cfg.base_sensor_range_m * 0.8,
                noise_std=0.18,
                update_rate_hz=0.9,
                tethered=False,
                mobile=False,
                active=True,
                metadata={"role": "valley_net"},
                trust=cfg.trust_init,
            )
        )
    sensors.append(
        SensorNodeNG(
            node_id="mp_patrol",
            kind="patrol_drone",
            modality=Modality.THERMAL,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.5,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.5,
            z=90.0,
            fov_deg=160.0,
            range_m=cfg.base_sensor_range_m * 2.0,
            noise_std=0.10,
            update_rate_hz=1.4,
            tethered=False,
            mobile=True,
            active=True,
            metadata={"role": "mountain_patrol"},
            trust=cfg.trust_init,
        )
    )
    sensors.append(
        SensorNodeNG(
            node_id="launcher_mountain",
            kind="launcher",
            modality=Modality.ACOUSTIC,
            x=cfg.grid_size_x * cfg.cell_size_m * 0.1,
            y=cfg.grid_size_y * cfg.cell_size_m * 0.9,
            z=1.0,
            fov_deg=0.0,
            range_m=0.0,
            noise_std=1.0,
            update_rate_hz=0.1,
            tethered=False,
            mobile=True,
            active=False,
            metadata={"role": "mountain_launcher"},
            trust=cfg.trust_init,
        )
    )
    return sensors


SENSOR_SUITES_NG: Dict[str, Callable[[SpatialConfigNG], List[SensorNodeNG]]] = {
    "mixed": suite_mixed,
    "dense_ground": suite_dense_ground,
    "sparse_long_range": suite_sparse_long_range,
    "forest_border": suite_forest_border,
    "coastal_port": suite_coastal_port,
    "industrial_dense": suite_industrial_dense,
    "mountain_pass_lr": suite_mountain_pass,
}


# ============================================================
#  DOMAIN EXTENSIONS (DESERT / ARCTIC)
# ============================================================

DOMAIN_CATALOG_NG["DESERT_TEST"] = DomainProfileNG(
    name="DESERT_TEST",
    scenario="open_field",
    sensor_suite="mixed",
    default_policy="coverage",
    description="High-visibility open desert test range with mixed airborne sensors.",
    subs={
        "baseline": {"policy": "coverage", "notes": "Baseline desert coverage."},
        "hot_sand": {"policy": "hotspot", "notes": "Localized heat / mirage-like hotspots."},
    },
)

DOMAIN_CATALOG_NG["ARCTIC_OUTPOST"] = DomainProfileNG(
    name="ARCTIC_OUTPOST",
    scenario="coastal",
    sensor_suite="coastal_port",
    default_policy="coverage",
    description="Arctic coastal outpost with ice/water contrast and harsh conditions.",
    subs={
        "ice_edge": {"policy": "perimeter", "notes": "Monitoring the ice/water boundary."},
        "harbor_safety": {"policy": "coverage", "notes": "Safe navigation and harbor status."},
    },
)


# ============================================================
#  DOMAIN CONTEXT / DIGITAL TWIN HOOKS
# ============================================================

def build_domain_context_ng(
    domain: str,
    sub: Optional[str] = None,
    override_cfg: Optional[Dict] = None,
) -> Tuple[SpatialConfigNG, Callable[[SpatialConfigNG], List[SensorNodeNG]], BasePolicyNG]:
    if domain not in DOMAIN_CATALOG_NG:
        raise ValueError(f"Unknown domain {domain}")
    profile = DOMAIN_CATALOG_NG[domain]

    cfg_kwargs: Dict[str, Any] = {}
    if override_cfg:
        cfg_kwargs.update(override_cfg)
    cfg = SpatialConfigNG(
        scenario=profile.scenario,
        **cfg_kwargs,
    )

    suite_key = profile.sensor_suite
    if suite_key not in SENSOR_SUITES_NG:
        raise ValueError(f"Unknown sensor suite {suite_key}")
    suite_fn = SENSOR_SUITES_NG[suite_key]

    policy_name = profile.default_policy
    if sub is not None and sub in profile.subs:
        policy_name = profile.subs[sub].get("policy", policy_name)

    policy = make_policy_ng(policy_name, cfg)
    return cfg, suite_fn, policy


def clone_cfg_for_what_if(cfg: SpatialConfigNG, **overrides) -> SpatialConfigNG:
    data = vars(cfg).copy()
    data.update(overrides)
    return SpatialConfigNG(**data)


# ============================================================
#  BENCHMARKS
# ============================================================

def run_single_benchmark_ng(
    domain: str,
    sub: str,
    cfg_template: SpatialConfigNG,
    suite_fn,
    policy: BasePolicyNG,
    seed: int,
    steps: int,
    verbose: bool = False,
) -> Dict[str, float]:
    cfg_local = clone_cfg_for_what_if(cfg_template, verbose=verbose)
    sensors = suite_fn(cfg_local)
    local_policy = make_policy_ng(policy.name, cfg_local)
    orch = OrchestratorNG(cfg_local, sensors, local_policy, seed=seed)
    t0 = time.time()
    orch.run(steps=steps)
    t1 = time.time()
    summary = orch.summarize()
    summary["runtime_s"] = t1 - t0
    summary["domain"] = domain
    summary["sub"] = sub
    summary["seed"] = seed
    return summary


def run_benchmarks_ng():
    print("============================================")
    print(" SPATIAL-OPSEC+INT NG · Benchmark Suite")
    print("============================================")
    pairs = [
        ("OPSEC_OPEN_FIELD", "global_coverage"),
        ("OPSEC_OPEN_FIELD", "balanced"),
        ("URBAN_SECURITY", "crowd_monitoring"),
        ("PERIMETER_MONITOR", "ring_coverage"),
        ("CRITICAL_FACILITY", "dense_coverage"),
        ("DISASTER_RESPONSE", "urban_search"),
        ("FOREST_BORDER", "deep_patrol"),
        ("COASTAL_PORT", "harbor_coverage"),
        ("MOUNTAIN_PASS", "pass_watch"),
    ]
    results: List[Dict[str, float]] = []
    for domain, sub in pairs:
        cfg0, suite_fn, policy = build_domain_context_ng(
            domain, sub, override_cfg={"verbose": False}
        )
        for i in range(cfg0.benchmark_runs):
            seed = cfg0.base_seed + i
            logger.info(
                f"[INFO] benchmark domain={domain} sub={sub} "
                f"run={i+1}/{cfg0.benchmark_runs} seed={seed} policy={policy.name}"
            )
            res = run_single_benchmark_ng(
                domain,
                sub,
                cfg0,
                suite_fn,
                policy,
                seed=seed,
                steps=cfg0.benchmark_steps,
                verbose=False,
            )
            results.append(res)

    by_key: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for r in results:
        key = (r["domain"], r["sub"])
        by_key.setdefault(key, []).append(r)

    print("\n================ SUMMARY (per domain/sub) ================")
    for (domain, sub), subset in by_key.items():
        def avg(field: str) -> float:
            return float(np.mean([x[field] for x in subset]))
        print(f"\nDomain/Sub: {domain} / {sub}")
        print(f"  coverage_mean:          {avg('coverage_mean'):.3f}")
        print(f"  coverage_final:         {avg('coverage_final'):.3f}")
        print(f"  confidence_mean:        {avg('confidence_mean'):.3f}")
        print(f"  confidence_final:       {avg('confidence_final'):.3f}")
        print(f"  env_corr_mean:          {avg('env_correlation_mean'):.3f}")
        print(f"  env_corr_final:         {avg('env_correlation_final'):.3f}")
        print(f"  entropy_mean_bits:      {avg('entropy_bits_mean'):.3f}")
        print(f"  entropy_final_bits:     {avg('entropy_bits_final'):.3f}")
        print(f"  dynamic_pods_final:     {avg('dynamic_pods_final'):.1f}")
        print(f"  dynamic_cost_final:     {avg('dynamic_cost_final'):.2f}")
        print(f"  env_corr_per_dyn_cost:  {avg('env_corr_per_dyn_cost'):.4f}")
        print(f"  total_cost_final:       {avg('total_asset_cost_final'):.2f}")
        print(f"  avg_runtime_s:          {avg('runtime_s'):.3f}")


# ============================================================
#  PRIVACY-AWARE TELEMETRY
# ============================================================

def make_privacy_mask(cfg: SpatialConfigNG, rng: np.random.Generator) -> np.ndarray:
    h, w = cfg.grid_size_y, cfg.grid_size_x
    mask = np.zeros((h, w), dtype=bool)
    num_cells = int(h * w * cfg.privacy_mask_fraction)
    xs = rng.integers(0, w, size=num_cells)
    ys = rng.integers(0, h, size=num_cells)
    mask[ys, xs] = True
    return mask


def export_telemetry_ng(
    domain: str,
    sub: str,
    orch: OrchestratorNG,
    apply_privacy: bool = True,
) -> Dict[str, Any]:
    cfg = orch.cfg
    rng = np.random.default_rng(cfg.base_seed + 999)
    privacy_mask = make_privacy_mask(cfg, rng) if apply_privacy else np.zeros_like(orch.env.activity, dtype=bool)

    activity = orch.fusion.activity_prob.copy()
    sem_probs = orch.fusion.semantic_probs.copy()

    if apply_privacy:
        noise = rng.normal(0.0, cfg.privacy_noise_sigma, size=activity.shape)
        activity = np.clip(activity + noise, 0.0, 1.0)
        activity[privacy_mask] = 0.0
        sem_probs[:, privacy_mask] = 1.0 / cfg.semantic_channels

    summary = orch.summarize()
    tel: Dict[str, Any] = {
        "domain": domain,
        "sub": sub,
        "cfg": vars(cfg),
        "policy": orch.assets.policy.name,
        "activity_prob": activity,
        "semantic_probs": sem_probs,
        "summary": summary,
        "privacy_mask": privacy_mask,
    }
    return tel


# ============================================================
#  SERIALIZATION HELPERS (JSON-SAFE)
# ============================================================

def _np_to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def serialize_telemetry_ng(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a telemetry dict to a JSON-safe structure.
    """
    out: Dict[str, Any] = {}
    for k, v in telemetry.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = serialize_telemetry_ng(v)  # recursive for nested dicts
        else:
            out[k] = v
    return out


def to_json_str(data: Any, indent: int = 2) -> str:
    """
    Convenience: serialize any engine output (telemetry, summary, records)
    to JSON string, converting numpy arrays as needed.
    """
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (SpatialConfigNG, DomainProfileNG)):
            return asdict(o)
        return str(o)
    return json.dumps(data, indent=indent, default=default)


# ============================================================
#  VISUALIZATIONS (BASE)
# ============================================================

def visualize_snapshot_ng(orch: OrchestratorNG, title: str = "Snapshot"):
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    env = orch.env
    fusion = orch.fusion
    sensors = orch.sensors

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    im0 = ax.imshow(env.activity, origin="lower")
    ax.set_title("Ground truth activity")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(fusion.activity_prob, origin="lower", vmin=0, vmax=1)
    ax.set_title("Belief P(activity)")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    sem_map = fusion.semantic_mode_map()
    ax.imshow(env.terrain, origin="lower", alpha=0.3)
    ax.imshow(sem_map, origin="lower", alpha=0.5)
    for s in sensors:
        gx, gy = s.world_to_grid(orch.cfg)
        if s.kind in ("drone", "patrol_drone"):
            marker = "v"
        elif s.kind in ("ground_pod", "forest_pod", "valley_pod"):
            marker = "s"
        elif s.kind in ("tower", "harbor_tower", "ridge_tower"):
            marker = "^"
        elif s.kind == "human":
            marker = "o"
        elif s.kind == "launched_pod":
            marker = "P"
        else:
            marker = "x"
        ax.scatter(gx, gy, marker=marker, s=40)
    ax.set_title("Semantic mode map + assets")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_timeseries_ng(orch: OrchestratorNG, title: str = "Run metrics"):
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    steps = np.arange(len(orch.metrics["coverage"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    ax = axes[0, 0]
    ax.plot(steps, orch.metrics["coverage"])
    ax.set_title("Coverage")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction")

    ax = axes[0, 1]
    ax.plot(steps, orch.metrics["env_correlation"])
    ax.set_title("Env correlation")
    ax.set_xlabel("step")
    ax.set_ylabel("corr")

    ax = axes[1, 0]
    ax.plot(steps, orch.metrics["entropy_bits"])
    ax.set_title("Entropy (bits)")
    ax.set_xlabel("step")
    ax.set_ylabel("bits")

    ax = axes[1, 1]
    ax.plot(steps, orch.metrics["dynamic_pods"], label="dyn pods")
    ax2 = ax.twinx()
    ax2.plot(steps, orch.metrics["dynamic_cost"], linestyle="--", label="dyn cost")
    ax.set_title("Dynamic pods / cost")
    ax.set_xlabel("step")
    ax.set_ylabel("# pods")
    ax2.set_ylabel("cost")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_domain_summary_ng(telemetry: List[Dict[str, Any]]):
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    labels = []
    env_corr_final = []
    cov_final = []
    for t in telemetry:
        labels.append(f"{t['domain'].split('_')[0]}-{t['sub']}")
        env_corr_final.append(t["summary"]["env_correlation_final"])
        cov_final.append(t["summary"]["coverage_final"])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.6), 6), sharex=True)

    ax = axes[0]
    ax.bar(x, env_corr_final)
    ax.set_ylabel("env_corr_final")
    ax.set_title("Env correlation by domain/sub")

    ax = axes[1]
    ax.bar(x, cov_final)
    ax.set_ylabel("coverage_final")
    ax.set_title("Coverage by domain/sub")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# ============================================================
#  DEMO / RUN HELPERS (BASE)
# ============================================================

def run_demo_ng():
    cfg, suite_fn, policy = build_domain_context_ng(
        "OPSEC_OPEN_FIELD",
        "balanced",
        override_cfg={"sim_steps": 200, "verbose": True},
    )
    sensors = suite_fn(cfg)
    orch = OrchestratorNG(cfg, sensors, policy, seed=cfg.base_seed)
    print("============================================")
    print(" SPATIAL-OPSEC+INT NG · Demo (OPSEC_OPEN_FIELD / balanced)")
    print("============================================")
    orch.run()
    summary = orch.summarize()
    print("\n=== DEMO SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.3f}")
    visualize_snapshot_ng(orch, title="OPSEC_OPEN_FIELD / balanced · final")
    visualize_timeseries_ng(orch, title="OPSEC_OPEN_FIELD / balanced · metrics")


def run_demo_urban_ng():
    cfg, suite_fn, policy = build_domain_context_ng(
        "URBAN_SECURITY",
        "crowd_monitoring",
        override_cfg={"sim_steps": 150, "verbose": True},
    )
    sensors = suite_fn(cfg)
    orch = OrchestratorNG(cfg, sensors, policy, seed=cfg.base_seed)
    print("============================================")
    print(" SPATIAL-OPSEC+INT NG · Demo (URBAN_SECURITY / crowd_monitoring)")
    print("============================================")
    orch.run()
    summary = orch.summarize()
    print("\n=== DEMO SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.3f}")
    visualize_snapshot_ng(orch, title="URBAN_SECURITY / crowd_monitoring · final")
    visualize_timeseries_ng(orch, title="URBAN_SECURITY / crowd_monitoring · metrics")


def run_all_domain_runs_ng() -> List[Dict[str, Any]]:
    print("============================================")
    print(" SPATIAL-OPSEC+INT NG · All Domain/Sub Runs")
    print("============================================")
    telemetry: List[Dict[str, Any]] = []
    for domain, profile in DOMAIN_CATALOG_NG.items():
        for sub in profile.subs.keys():
            cfg, suite_fn, policy = build_domain_context_ng(
                domain, sub, override_cfg={"verbose": False}
            )
            sensors = suite_fn(cfg)
            orch = OrchestratorNG(cfg, sensors, policy, seed=cfg.base_seed)
            t0 = time.time()
            orch.run()
            t1 = time.time()
            summary = orch.summarize()
            print(
                f"\n--- Domain={domain} Sub={sub} Policy={policy.name} "
                f"Runtime={t1 - t0:.3f}s ---"
            )
            for k, v in summary.items():
                print(f" {k:30s}: {v:.3f}")
            tel = export_telemetry_ng(domain, sub, orch, apply_privacy=True)
            tel["runtime_s"] = t1 - t0
            telemetry.append(tel)
    return telemetry


def run_policy_sweep_ng(
    domain: str,
    sub: str,
    policies=("coverage", "hotspot", "perimeter", "hybrid"),
    pod_options=(4, 8, 16),
    deploy_intervals=(10, 15, 20),
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    print("============================================")
    print(f" SPATIAL-OPSEC+INT NG · Policy Sweep [{domain}/{sub}]")
    print("============================================")
    rows: List[Dict[str, Any]] = []
    for pol in policies:
        for max_pods in pod_options:
            for interval in deploy_intervals:
                cfg, suite_fn, _ = build_domain_context_ng(
                    domain,
                    sub,
                    override_cfg={
                        "max_dynamic_pods": max_pods,
                        "deploy_interval_steps": interval,
                        "verbose": verbose,
                    },
                )
                sensors = suite_fn(cfg)
                policy = make_policy_ng(pol, cfg)
                orch = OrchestratorNG(cfg, sensors, policy, seed=cfg.base_seed)
                t0 = time.time()
                orch.run()
                t1 = time.time()
                summary = orch.summarize()
                row = {
                    "policy": pol,
                    "max_pods": max_pods,
                    "interval": interval,
                    "cov_final": summary["coverage_final"],
                    "env_corr_final": summary["env_correlation_final"],
                    "entropy_final_bits": summary["entropy_bits_final"],
                    "dynamic_pods_final": summary["dynamic_pods_final"],
                    "dynamic_cost_final": summary["dynamic_cost_final"],
                    "env_corr_per_dyn_cost": summary["env_corr_per_dyn_cost"],
                    "total_cost_final": summary["total_asset_cost_final"],
                    "runtime_s": t1 - t0,
                }
                rows.append(row)
    header = (
        "policy   max_pods  interval  cov_final  env_corr_final  "
        "H_final(bits)  dyn_pods_final  dyn_cost  corr/dyn_cost  total_cost  runtime_s"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['policy']:8s} "
            f"{r['max_pods']:8d} "
            f"{r['interval']:8d} "
            f"{r['cov_final']:10.3f} "
            f"{r['env_corr_final']:15.3f} "
            f"{r['entropy_final_bits']:13.3f} "
            f"{int(r['dynamic_pods_final']):15d} "
            f"{r['dynamic_cost_final']:8.2f} "
            f"{r['env_corr_per_dyn_cost']:12.4f} "
            f"{r['total_cost_final']:11.2f} "
            f"{r['runtime_s']:9.3f}"
        )
    return rows


# ============================================================
#  RISK / ERROR ANALYZER
# ============================================================

@dataclass
class RiskSummaryNG:
    mean_abs_error: float
    max_abs_error: float
    blindspot_risk: float
    overconfidence_risk: float
    underconfidence_risk: float
    cost_efficiency: float
    env_corr_final: float
    coverage_final: float
    entropy_final_bits: float


class RiskAnalyzerNG:
    """
    Offline analyzer using ground-truth access inside the simulator.
    Not available in field, but critical for R&D.
    """

    def __init__(self, orch: OrchestratorNG):
        self.orch = orch
        self.cfg = orch.cfg
        self.env = orch.env
        self.fusion = orch.fusion
        self.metrics = orch.metrics

    def compute_error_fields(self) -> Dict[str, np.ndarray]:
        gt = self.env.activity
        est = self.fusion.activity_prob
        abs_err = np.abs(gt - est)
        sq_err = (gt - est) ** 2
        return {
            "abs_error": abs_err,
            "sq_error": sq_err,
            "gt": gt,
            "est": est,
        }

    def compute_risks(self) -> RiskSummaryNG:
        err = self.compute_error_fields()
        abs_err = err["abs_error"]
        gt = err["gt"]
        est = err["est"]

        mean_abs_error = float(abs_err.mean())
        max_abs_error = float(abs_err.max())

        high_gt = gt > 0.6
        low_est = est < 0.3
        blindspot_mask = high_gt & low_est
        blindspot_risk = float(abs_err[blindspot_mask].mean()) if blindspot_mask.any() else 0.0

        high_est = est > 0.7
        low_gt = gt < 0.2
        over_mask = high_est & low_gt
        overconfidence_risk = float(abs_err[over_mask].mean()) if over_mask.any() else 0.0

        medium_est = (est > 0.3) & (est < 0.7)
        under_mask = medium_est & high_gt
        underconfidence_risk = float(abs_err[under_mask].mean()) if under_mask.any() else 0.0

        summary = self.orch.summarize()
        env_corr_final = summary["env_correlation_final"]
        total_cost_final = summary["total_asset_cost_final"]
        cost_eff = env_corr_final / max(total_cost_final, 1e-6)

        return RiskSummaryNG(
            mean_abs_error=mean_abs_error,
            max_abs_error=max_abs_error,
            blindspot_risk=blindspot_risk,
            overconfidence_risk=overconfidence_risk,
            underconfidence_risk=underconfidence_risk,
            cost_efficiency=cost_eff,
            env_corr_final=env_corr_final,
            coverage_final=summary["coverage_final"],
            entropy_final_bits=summary["entropy_bits_final"],
        )


# ============================================================
#  SENSOR-LEVEL TELEMETRY / TRUST ANALYSIS
# ============================================================

def compute_sensor_telemetry_ng(orch: OrchestratorNG) -> List[Dict[str, Any]]:
    """
    Per-sensor telemetry:
    - final trust
    - modality, kind
    - approximate signal energy (if last_obs exists)
    """
    records: List[Dict[str, Any]] = []
    for s in orch.sensors:
        rec: Dict[str, Any] = {
            "node_id": s.node_id,
            "kind": s.kind,
            "modality": s.modality,
            "trust": float(getattr(s, "trust", 0.0)),
        }
        if s.last_obs is not None:
            energy = float(np.mean(np.abs(s.last_obs)))
        else:
            energy = 0.0
        rec["signal_energy"] = energy
        records.append(rec)
    return records


# ============================================================
#  ADDITIONAL VISUALIZATIONS
# ============================================================

def visualize_error_and_risk_ng(orch: OrchestratorNG, title: str = "Error / risk maps"):
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    ra = RiskAnalyzerNG(orch)
    fields = ra.compute_error_fields()
    abs_err = fields["abs_error"]
    gt = fields["gt"]
    est = fields["est"]

    high_gt = gt > 0.6
    low_est = est < 0.3
    blindspot = np.zeros_like(gt)
    blindspot[high_gt & low_est] = 1.0

    high_est = est > 0.7
    low_gt = gt < 0.2
    overconf = np.zeros_like(gt)
    overconf[high_est & low_gt] = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    im0 = ax.imshow(abs_err, origin="lower")
    ax.set_title("Absolute error |gt - est|")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(blindspot, origin="lower", vmin=0, vmax=1)
    ax.set_title("Blindspot zones (gt high, est low)")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im2 = ax.imshow(overconf, origin="lower", vmin=0, vmax=1)
    ax.set_title("Overconfidence zones (est high, gt low)")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_sensor_trust_ng(orch: OrchestratorNG, title: str = "Sensor trust by modality"):
    if not HAS_MPL:
        print("Matplotlib not available.")
        return
    records = compute_sensor_telemetry_ng(orch)
    if not records:
        print("No sensor telemetry available.")
        return

    modalities = sorted({r["modality"] for r in records})
    avg_trust = []
    avg_energy = []
    for m in modalities:
        group = [r for r in records if r["modality"] == m]
        avg_trust.append(float(np.mean([g["trust"] for g in group])))
        avg_energy.append(float(np.mean([g["signal_energy"] for g in group])))

    x = np.arange(len(modalities))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.bar(x, avg_trust)
    ax.set_ylabel("avg trust")
    ax.set_title("Average sensor trust by modality")

    ax = axes[1]
    ax.bar(x, avg_energy)
    ax.set_ylabel("avg |signal|")
    ax.set_title("Average signal energy by modality")
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=20, ha="right")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
#  EXPERIMENT MANAGER · LARGE PARAM SWEEPS
# ============================================================

class ExperimentManagerNG:
    """
    Coordinate multi-domain / multi-policy experiments with
    consistent reporting.
    """

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def run_single(
        self,
        domain: str,
        sub: str,
        policy_name: Optional[str] = None,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        cfg, suite_fn, default_policy = build_domain_context_ng(
            domain, sub, override_cfg=cfg_overrides or {}
        )
        if steps is not None:
            cfg.sim_steps = steps
        if not verbose:
            cfg.verbose = False

        policy = make_policy_ng(policy_name or default_policy.name, cfg)
        sensors = suite_fn(cfg)
        orch = OrchestratorNG(cfg, sensors, policy, seed=seed or cfg.base_seed)
        t0 = time.time()
        orch.run()
        t1 = time.time()
        summary = orch.summarize()
        risk = RiskAnalyzerNG(orch).compute_risks() if cfg.enable_risk_analysis else None
        rec: Dict[str, Any] = {
            "domain": domain,
            "sub": sub,
            "policy": policy.name,
            "sim_steps": cfg.sim_steps,
            "seed": seed or cfg.base_seed,
            "runtime_s": t1 - t0,
            "coverage_final": summary["coverage_final"],
            "env_corr_final": summary["env_correlation_final"],
            "entropy_final_bits": summary["entropy_bits_final"],
            "total_cost_final": summary["total_asset_cost_final"],
        }
        if risk is not None:
            rec.update(
                {
                    "mean_abs_error": risk.mean_abs_error,
                    "blindspot_risk": risk.blindspot_risk,
                    "overconfidence_risk": risk.overconfidence_risk,
                    "underconfidence_risk": risk.underconfidence_risk,
                    "cost_efficiency": risk.cost_efficiency,
                }
            )
        self.rows.append(rec)
        return {
            "orch": orch,
            "summary": summary,
            "risk": risk,
            "record": rec,
        }

    def run_grid(
        self,
        domains: List[str],
        subs_by_domain: Dict[str, List[str]],
        policies: List[str],
        seeds: List[int],
        cfg_overrides: Optional[Dict[str, Any]] = None,
        steps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        print("============================================")
        print(" SPATIAL-OPSEC+INT NG · Experiment Grid")
        print("============================================")
        all_records: List[Dict[str, Any]] = []
        for domain in domains:
            subs = subs_by_domain.get(domain, list(DOMAIN_CATALOG_NG[domain].subs.keys()))
            for sub in subs:
                for pol in policies:
                    for seed in seeds:
                        logger.info(f"[EXP] domain={domain} sub={sub} policy={pol} seed={seed}")
                        out = self.run_single(
                            domain=domain,
                            sub=sub,
                            policy_name=pol,
                            cfg_overrides=cfg_overrides,
                            seed=seed,
                            steps=steps,
                            verbose=False,
                        )
                        all_records.append(out["record"])
        return all_records


# ============================================================
#  TEXT REPORT GENERATOR · PER DOMAIN
# ============================================================

def generate_text_report_ng(telemetry: List[Dict[str, Any]]):
    """
    Produces a concise multi-domain textual report suitable for
    copy-paste into a technical memo.
    """
    print("====================================================")
    print(" SPATIAL-OPSEC+INT NG · Multi-Domain Text Report")
    print("====================================================\n")
    for t in telemetry:
        domain = t["domain"]
        sub = t["sub"]
        summary = t["summary"]
        print(f"[Domain: {domain}  /  Sub-scenario: {sub}]")
        print(f"  Policy:               {t['policy']}")
        print(f"  coverage_mean:        {summary['coverage_mean']:.3f}")
        print(f"  coverage_final:       {summary['coverage_final']:.3f}")
        print(f"  env_corr_mean:        {summary['env_correlation_mean']:.3f}")
        print(f"  env_corr_final:       {summary['env_correlation_final']:.3f}")
        print(f"  entropy_mean_bits:    {summary['entropy_bits_mean']:.3f}")
        print(f"  entropy_final_bits:   {summary['entropy_bits_final']:.3f}")
        print(f"  static_asset_cost:    {summary['static_asset_cost']:.2f}")
        print(f"  dynamic_cost_final:   {summary['dynamic_cost_final']:.2f}")
        print(f"  total_cost_final:     {summary['total_asset_cost_final']:.2f}")
        print(f"  env_corr_per_dyn_cost:{summary['env_corr_per_dyn_cost']:.4f}")
        print("")


# ============================================================
#  APPLICATION CATALOG · ALL DOMAINS + ALL PARTS
# ============================================================

@dataclass
class ApplicationProfileNG:
    """
    High-level application domain:
    - id: stable key (e.g. "MESH_NETWORKING")
    - label: human-readable name
    - description: what this application represents
    - domain_subs: list of (domain, sub) tuples that instantiate this app
    - default_policies: optional per (domain,sub) override of policy
    """
    app_id: str
    label: str
    description: str
    domain_subs: List[Tuple[str, str]]
    default_policies: Dict[Tuple[str, str], str] = field(default_factory=dict)


APPLICATION_CATALOG_NG: Dict[str, ApplicationProfileNG] = {
    "MESH_NETWORKING": ApplicationProfileNG(
        app_id="MESH_NETWORKING",
        label="Mesh Networking & Off-Grid Comms",
        description=(
            "Routing, coverage and robustness for off-grid mesh networks, "
            "disaster comms and long-range perimeter connectivity."
        ),
        domain_subs=[
            ("OPSEC_OPEN_FIELD", "global_coverage"),
            ("FOREST_BORDER", "border_watch"),
            ("MOUNTAIN_PASS", "pass_watch"),
            ("DESERT_TEST", "baseline"),
            ("ARCTIC_OUTPOST", "ice_edge"),
        ],
        default_policies={
            ("FOREST_BORDER", "border_watch"): "perimeter",
            ("MOUNTAIN_PASS", "pass_watch"): "perimeter",
        },
    ),
    "ROBOTICS_SWARM": ApplicationProfileNG(
        app_id="ROBOTICS_SWARM",
        label="Robotics & Swarm Coordination",
        description=(
            "Stability-aware control and situational awareness for multi-robot "
            "and drone swarms in structured and unstructured environments."
        ),
        domain_subs=[
            ("URBAN_SECURITY", "crowd_monitoring"),
            ("CRITICAL_FACILITY", "dense_coverage"),
            ("OPSEC_OPEN_FIELD", "hotspot_tracking"),
        ],
        default_policies={
            ("URBAN_SECURITY", "crowd_monitoring"): "hotspot",
            ("OPSEC_OPEN_FIELD", "hotspot_tracking"): "hotspot",
        },
    ),
    "SMART_CITY": ApplicationProfileNG(
        app_id="SMART_CITY",
        label="Smart City & Civic Sensing",
        description=(
            "Traffic, crowd and infrastructure sensing for smart city operations, "
            "without direct control over actuators."
        ),
        domain_subs=[
            ("URBAN_SECURITY", "infrastructure_monitoring"),
            ("DISASTER_RESPONSE", "urban_search"),
            ("COASTAL_PORT", "harbor_coverage"),
        ],
        default_policies={},
    ),
    "LOGISTICS_FLOW": ApplicationProfileNG(
        app_id="LOGISTICS_FLOW",
        label="Logistics & Supply Flow Digital Twin",
        description=(
            "Abstracted representation of ports, warehouses, perimeters and "
            "transport corridors for flow optimization and resilience testing."
        ),
        domain_subs=[
            ("COASTAL_PORT", "channel_focus"),
            ("CRITICAL_FACILITY", "fallback"),
            ("PERIMETER_MONITOR", "ring_coverage"),
        ],
        default_policies={},
    ),
    "ENERGY_GRID": ApplicationProfileNG(
        app_id="ENERGY_GRID",
        label="Energy / Utilities Stability (Abstracted)",
        description=(
            "Uses spatial fields as a proxy for load/flow distribution and "
            "sensor reliability in abstracted grids and networks."
        ),
        domain_subs=[
            ("OPSEC_OPEN_FIELD", "balanced"),
            ("PERIMETER_MONITOR", "breach_focus"),
        ],
        default_policies={},
    ),
    "MEDICAL_NEURO": ApplicationProfileNG(
        app_id="MEDICAL_NEURO",
        label="Neuro-inspired Network Dynamics (R&D)",
        description=(
            "R&D-only application modeling abstracted brain-region coherence, "
            "synchronization and blindspots using spatial fields."
        ),
        domain_subs=[
            ("URBAN_SECURITY", "hotspot_focus"),
            ("FOREST_BORDER", "deep_patrol"),
        ],
        default_policies={},
    ),
    "FINANCE_EXECUTION": ApplicationProfileNG(
        app_id="FINANCE_EXECUTION",
        label="Execution & Liquidity Field (Abstracted)",
        description=(
            "Uses open-field activity as a proxy for liquidity hotspots and "
            "perimeter dynamics as risk/constraint boundaries."
        ),
        domain_subs=[
            ("OPSEC_OPEN_FIELD", "hotspot_tracking"),
            ("PERIMETER_MONITOR", "breach_focus"),
        ],
        default_policies={},
    ),
    "HUMAN_SYSTEMS": ApplicationProfileNG(
        app_id="HUMAN_SYSTEMS",
        label="Human Systems & Group Dynamics (R&D)",
        description=(
            "Abstracted group, crowd and organizational behaviors for risk, "
            "blindspot and overconfidence studies."
        ),
        domain_subs=[
            ("URBAN_SECURITY", "crowd_monitoring"),
            ("DISASTER_RESPONSE", "perimeter_triage"),
            ("ARCTIC_OUTPOST", "harbor_safety"),
        ],
        default_policies={},
    ),
}


class ApplicationExperimentRunnerNG:
    """
    Runs experiments at the "application" level:
    - loops over all (domain, sub) combinations for that app
    - applies default or specified policies
    - returns a consolidated report
    """

    def __init__(self, app_id: str):
        if app_id not in APPLICATION_CATALOG_NG:
            raise ValueError(f"Unknown application id {app_id}")
        self.profile = APPLICATION_CATALOG_NG[app_id]
        self.exp_mgr = ExperimentManagerNG()

    def run(
        self,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        seeds = seeds or [17]
        all_records: List[Dict[str, Any]] = []
        last_och_by_ds: Dict[Tuple[str, str], OrchestratorNG] = {}
        print("====================================================")
        print(f" Application Run: {self.profile.app_id} · {self.profile.label}")
        print("====================================================")
        print(self.profile.description)
        print("----------------------------------------------------")

        for (domain, sub) in self.profile.domain_subs:
            for seed in seeds:
                pol_override = self.profile.default_policies.get((domain, sub), None)
                logger.info(
                    f"[APP] app={self.profile.app_id} domain={domain} sub={sub} "
                    f"policy={pol_override or 'default'} seed={seed}"
                )
                out = self.exp_mgr.run_single(
                    domain=domain,
                    sub=sub,
                    policy_name=pol_override,
                    cfg_overrides=cfg_overrides,
                    seed=seed,
                    steps=steps,
                    verbose=False,
                )
                all_records.append(out["record"])
                last_och_by_ds[(domain, sub)] = out["orch"]

        def agg(field: str) -> float:
            vals = [r[field] for r in all_records if field in r]
            return float(np.mean(vals)) if vals else 0.0

        summary: Dict[str, Any] = {
            "app_id": self.profile.app_id,
            "label": self.profile.label,
            "description": self.profile.description,
            "num_runs": len(all_records),
            "coverage_final_mean": agg("coverage_final"),
            "env_corr_final_mean": agg("env_corr_final"),
            "entropy_final_bits_mean": agg("entropy_final_bits"),
            "total_cost_final_mean": agg("total_cost_final"),
        }
        if "mean_abs_error" in all_records[0]:
            summary.update(
                {
                    "mean_abs_error_mean": agg("mean_abs_error"),
                    "blindspot_risk_mean": agg("blindspot_risk"),
                    "overconfidence_risk_mean": agg("overconfidence_risk"),
                    "underconfidence_risk_mean": agg("underconfidence_risk"),
                    "cost_efficiency_mean": agg("cost_efficiency"),
                }
            )

        print("\n================ APPLICATION SUMMARY ================")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k:30s}: {v:.4f}")
            else:
                print(f"{k:30s}: {v}")
        print("====================================================")

        return {
            "summary": summary,
            "records": all_records,
            "last_orchestrators": last_och_by_ds,
        }


def generate_application_text_report_ng(app_result: Dict[str, Any]):
    """
    Compact report focusing on the application-level view.
    """
    summary = app_result["summary"]
    print("====================================================")
    print(f" SPATIAL-OPSEC+INT NG · Application Report: {summary['app_id']}")
    print("====================================================")
    print(f"Label       : {summary['label']}")
    print(f"Description : {summary['description']}")
    print(f"Num Runs    : {summary['num_runs']}")
    print("")
    fields = [
        "coverage_final_mean",
        "env_corr_final_mean",
        "entropy_final_bits_mean",
        "total_cost_final_mean",
        "mean_abs_error_mean",
        "blindspot_risk_mean",
        "overconfidence_risk_mean",
        "underconfidence_risk_mean",
        "cost_efficiency_mean",
    ]
    for f in fields:
        if f in summary:
            print(f"{f:30s}: {summary[f]:.4f}")
    print("====================================================")


# ============================================================
#  POLICY OPTIMIZER (NEW)
# ============================================================

@dataclass
class PolicyOptimizerResultNG:
    domain: str
    sub: str
    metric_name: str
    best_metric: float
    best_policy: str
    best_max_pods: int
    best_interval: int
    table: List[Dict[str, Any]]


class PolicyOptimizerNG:
    """
    Lightweight optimizer for deployment policy hyperparameters for a given
    (domain, sub) pair.

    It sweeps:
        policies × max_dynamic_pods × deploy_interval_steps
    and picks the combination maximizing a chosen metric (default: env_corr_per_dyn_cost).
    """

    def __init__(
        self,
        domain: str,
        sub: str,
        policies: Tuple[str, ...] = ("coverage", "hotspot", "perimeter", "hybrid"),
        pod_options: Tuple[int, ...] = (4, 8, 16),
        deploy_intervals: Tuple[int, ...] = (10, 15, 20),
        metric: str = "env_corr_per_dyn_cost",
    ):
        self.domain = domain
        self.sub = sub
        self.policies = policies
        self.pod_options = pod_options
        self.deploy_intervals = deploy_intervals
        self.metric = metric

    def optimize(self, verbose: bool = False) -> PolicyOptimizerResultNG:
        rows = run_policy_sweep_ng(
            self.domain,
            self.sub,
            policies=self.policies,
            pod_options=self.pod_options,
            deploy_intervals=self.deploy_intervals,
            verbose=verbose,
        )
        if not rows:
            raise RuntimeError("Policy sweep produced no rows")
        best_row = max(rows, key=lambda r: r.get(self.metric, float("-inf")))
        return PolicyOptimizerResultNG(
            domain=self.domain,
            sub=self.sub,
            metric_name=self.metric,
            best_metric=best_row[self.metric],
            best_policy=best_row["policy"],
            best_max_pods=best_row["max_pods"],
            best_interval=best_row["interval"],
            table=rows,
        )


# ============================================================
#  PUSHED DEMO
# ============================================================

def run_pushed_demo_ng():
    """
    High-intensity demo that:
    - runs an URBAN_SECURITY / crowd_monitoring scenario,
    - computes risk metrics,
    - visualizes error/risk maps and sensor trust.
    """
    cfg, suite_fn, policy = build_domain_context_ng(
        "URBAN_SECURITY",
        "crowd_monitoring",
        override_cfg={"sim_steps": 180, "verbose": True},
    )
    sensors = suite_fn(cfg)
    orch = OrchestratorNG(cfg, sensors, policy, seed=cfg.base_seed)
    print("============================================")
    print(" SPATIAL-OPSEC+INT NG · Pushed Demo (URBAN / crowd)")
    print("============================================")
    orch.run()
    summary = orch.summarize()
    ra = RiskAnalyzerNG(orch)
    risk = ra.compute_risks()

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.3f}")

    print("\n=== RISK SUMMARY ===")
    print(f"mean_abs_error        : {risk.mean_abs_error:.3f}")
    print(f"max_abs_error         : {risk.max_abs_error:.3f}")
    print(f"blindspot_risk        : {risk.blindspot_risk:.3f}")
    print(f"overconfidence_risk   : {risk.overconfidence_risk:.3f}")
    print(f"underconfidence_risk  : {risk.underconfidence_risk:.3f}")
    print(f"cost_efficiency       : {risk.cost_efficiency:.5f}")
    print(f"env_corr_final        : {risk.env_corr_final:.3f}")
    print(f"coverage_final        : {risk.coverage_final:.3f}")
    print(f"entropy_final_bits    : {risk.entropy_final_bits:.3f}")

    visualize_snapshot_ng(orch, title="URBAN crowd_monitoring · final")
    visualize_error_and_risk_ng(orch, title="URBAN crowd_monitoring · error / risk")
    visualize_sensor_trust_ng(orch, title="URBAN crowd_monitoring · sensor trust")


# ============================================================
#  ENGINE FACADE (NEW) · CLEAN API SURFACE
# ============================================================

class EngineFacadeNG:
    """
    High-level interface around the engine for services, CLIs, and notebooks.

    Typical use:
        engine = EngineFacadeNG()
        domains = engine.list_domains()
        subs = engine.list_subs("URBAN_SECURITY")
        result = engine.run_domain("URBAN_SECURITY", "crowd_monitoring")
        app = engine.run_application("MESH_NETWORKING")
        opt = engine.optimize_policy("FOREST_BORDER", "deep_patrol")
    """

    def list_domains(self) -> List[str]:
        return sorted(DOMAIN_CATALOG_NG.keys())

    def describe_domain(self, domain: str) -> Dict[str, Any]:
        if domain not in DOMAIN_CATALOG_NG:
            raise ValueError(f"Unknown domain {domain}")
        profile = DOMAIN_CATALOG_NG[domain]
        return {
            "name": profile.name,
            "scenario": profile.scenario,
            "sensor_suite": profile.sensor_suite,
            "default_policy": profile.default_policy,
            "description": profile.description,
            "subs": list(profile.subs.keys()),
        }

    def list_subs(self, domain: str) -> List[str]:
        return self.describe_domain(domain)["subs"]

    def list_applications(self) -> List[str]:
        return sorted(APPLICATION_CATALOG_NG.keys())

    def describe_application(self, app_id: str) -> Dict[str, Any]:
        if app_id not in APPLICATION_CATALOG_NG:
            raise ValueError(f"Unknown application {app_id}")
        profile = APPLICATION_CATALOG_NG[app_id]
        return {
            "app_id": profile.app_id,
            "label": profile.label,
            "description": profile.description,
            "domain_subs": profile.domain_subs,
        }

    def run_domain(
        self,
        domain: str,
        sub: str,
        policy: Optional[str] = None,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        include_telemetry: bool = False,
    ) -> Dict[str, Any]:
        mgr = ExperimentManagerNG()
        out = mgr.run_single(
            domain=domain,
            sub=sub,
            policy_name=policy,
            cfg_overrides=cfg_overrides,
            seed=seed,
            steps=steps,
            verbose=False,
        )
        orch: OrchestratorNG = out["orch"]
        summary = out["summary"]
        risk = out["risk"]
        result: Dict[str, Any] = {
            "domain": domain,
            "sub": sub,
            "policy": out["record"]["policy"],
            "summary": summary,
        }
        if risk is not None:
            result["risk"] = asdict(risk)
        if include_telemetry:
            tel = export_telemetry_ng(domain, sub, orch, apply_privacy=True)
            result["telemetry"] = serialize_telemetry_ng(tel)
        return result

    def run_application(
        self,
        app_id: str,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        runner = ApplicationExperimentRunnerNG(app_id)
        result = runner.run(
            cfg_overrides=cfg_overrides,
            seeds=seeds,
            steps=steps,
        )
        return {
            "summary": result["summary"],
            "records": result["records"],
        }

    def optimize_policy(
        self,
        domain: str,
        sub: str,
        metric: str = "env_corr_per_dyn_cost",
        policies: Tuple[str, ...] = ("coverage", "hotspot", "perimeter", "hybrid"),
        pod_options: Tuple[int, ...] = (4, 8, 16),
        deploy_intervals: Tuple[int, ...] = (10, 15, 20),
        verbose: bool = False,
    ) -> Dict[str, Any]:
        optimizer = PolicyOptimizerNG(
            domain=domain,
            sub=sub,
            policies=policies,
            pod_options=pod_options,
            deploy_intervals=deploy_intervals,
            metric=metric,
        )
        result = optimizer.optimize(verbose=verbose)
        return {
            "domain": result.domain,
            "sub": result.sub,
            "metric_name": result.metric_name,
            "best_metric": result.best_metric,
            "best_policy": result.best_policy,
            "best_max_pods": result.best_max_pods,
            "best_interval": result.best_interval,
            "table": result.table,
        }


# ============================================================
#  SIMPLE DISPATCH / MAIN
# ============================================================

def main(profile: str = "full"):
    """
    profile:
      - "demo"       : run open-field + urban demos
      - "domains"    : run all domain/sub combos
      - "benchmarks" : run benchmark suite
      - "pushed"     : run pushed urban demo with risk viz
      - "app_mesh"   : run MESH_NETWORKING application
      - "app_all"    : run all applications (lightweight seeds)
      - "optimize"   : run a sample policy optimizer
      - "full"       : demos + domains + one app + optimizer (default)
    """
    if profile in ("demo", "full"):
        run_demo_ng()
        run_demo_urban_ng()

    if profile in ("domains", "full"):
        telemetry = run_all_domain_runs_ng()
        print(f"\n[INFO] collected telemetry records: {len(telemetry)}")
        if HAS_MPL:
            visualize_domain_summary_ng(telemetry)
        simple_tel = []
        for t in telemetry:
            simple_tel.append(
                {
                    "domain": t["domain"],
                    "sub": t["sub"],
                    "policy": t["policy"],
                    "summary": t["summary"],
                }
            )
        generate_text_report_ng(simple_tel)

    if profile in ("benchmarks", "full"):
        run_benchmarks_ng()

    if profile in ("pushed", "full"):
        run_pushed_demo_ng()

    if profile in ("app_mesh", "app_all", "full"):
        app_runner = ApplicationExperimentRunnerNG("MESH_NETWORKING")
        app_result = app_runner.run(
            cfg_overrides={"verbose": False, "sim_steps": 160},
            seeds=[17, 18],
        )
        generate_application_text_report_ng(app_result)

    if profile == "app_all":
        for app_id in APPLICATION_CATALOG_NG.keys():
            if app_id == "MESH_NETWORKING":
                continue
            app_runner = ApplicationExperimentRunnerNG(app_id)
            app_result = app_runner.run(
                cfg_overrides={"verbose": False, "sim_steps": 140},
                seeds=[17],
            )
            generate_application_text_report_ng(app_result)

    if profile in ("optimize", "full"):
        print("\n============================================")
        print(" Policy Optimizer Demo (FOREST_BORDER / deep_patrol)")
        print("============================================")
        engine = EngineFacadeNG()
        opt_result = engine.optimize_policy(
            "FOREST_BORDER",
            "deep_patrol",
            metric="env_corr_per_dyn_cost",
            policies=("coverage", "hotspot", "perimeter", "hybrid"),
            pod_options=(4, 8, 16),
            deploy_intervals=(8, 12, 16),
            verbose=False,
        )
        print("Best policy configuration:")
        print(to_json_str(opt_result, indent=2))


if __name__ == "__main__":
    logger.info("SPATIAL-OPSEC+INT NG v3.1 · Production-Oriented Engine")
    main(profile="full")
