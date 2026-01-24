#!/usr/bin/env python3
# ======================================================================================
# MULTI_FOV_CONTROLLER · v2
# Multi-point calibration schedules per κ=(i,p,b) with linear OR mildly-nonlinear fits
# Deterministic metadata · drift validation per band · simulation-ready
# ======================================================================================

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------- Core Types ---------------------------------------------

Band = str      # e.g. "b1", "b2"
Path = str      # e.g. "p1", "p2"


@dataclass(frozen=True)
class ConfigKey:
    """κ = (i,p,b)"""
    i: int
    p: Path
    b: Band

    def to_dict(self) -> Dict[str, Any]:
        return {"i": self.i, "p": self.p, "b": self.b}


@dataclass
class PoseSE3:
    """T = [R t; 0 1] in SE(3)."""
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


@dataclass
class BandIntrinsics:
    """Host camera intrinsics per band."""
    K: np.ndarray
    dist: np.ndarray
    resolution: Tuple[int, int]  # (W,H)


# ----------------------------- Phi Models ---------------------------------------------

class PhiModel:
    def id(self) -> str:
        raise NotImplementedError

    def apply_direction(self, u_hat: np.ndarray, i: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class FirstOrderAfocalPhi(PhiModel):
    phi_name: str
    magnification: float
    axis_by_i: Dict[int, np.ndarray]  # i -> axis (3,)

    def id(self) -> str:
        return f"phi:first_order:{self.phi_name}:M={self.magnification:.6g}"

    def apply_direction(self, u_hat: np.ndarray, i: int) -> np.ndarray:
        a = self.axis_by_i[i]
        a = a / (np.linalg.norm(a) + 1e-12)
        M = float(self.magnification)
        u = u_hat / (np.linalg.norm(u_hat) + 1e-12)
        transverse = u - (np.dot(u, a) * a)
        out = a + (1.0 / M) * transverse
        out = out / (np.linalg.norm(out) + 1e-12)
        return out


# ----------------------------- Radiometry Models --------------------------------------

class RadiometricModelKind(str, enum.Enum):
    LINEAR = "linear"          # y = a*L + c
    QUADRATIC = "quadratic"    # y = a*L^2 + b*L + c


@dataclass
class RadiometricFit:
    """
    Per κ model for mapping known calibration radiance L -> expected mean pixel y.

    LINEAR:      y = a*L + c         coeffs = [a, c]
    QUADRATIC:   y = a*L^2 + b*L + c coeffs = [a, b, c]
    """
    kind: RadiometricModelKind
    coeffs: np.ndarray
    L_domain: Tuple[float, float]         # (minL, maxL) used during fit
    rms: float                             # fit error over calibration points (mean ROI)
    updated_unix_s: float
    n_points: int
    temperatures_K: List[float]            # used
    notes: Dict[str, Any]

    def predict(self, L: np.ndarray) -> np.ndarray:
        L = np.asarray(L, dtype=float)
        if self.kind == RadiometricModelKind.LINEAR:
            a, c = self.coeffs.tolist()
            return a * L + c
        elif self.kind == RadiometricModelKind.QUADRATIC:
            a, b, c = self.coeffs.tolist()
            return a * (L ** 2) + b * L + c
        else:
            raise ValueError(f"Unknown radiometric kind: {self.kind}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "coeffs": self.coeffs.tolist(),
            "L_domain": [float(self.L_domain[0]), float(self.L_domain[1])],
            "rms": float(self.rms),
            "updated_unix_s": float(self.updated_unix_s),
            "n_points": int(self.n_points),
            "temperatures_K": [float(t) for t in self.temperatures_K],
            "notes": dict(self.notes),
        }


@dataclass
class DriftStats:
    """
    Drift statistics computed against the current RadiometricFit for calibration ROI.
    We store residual mean/variance and a flag.
    """
    mu: float
    sigma2: float
    n: int
    updated_unix_s: float
    flagged: bool
    detail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu": float(self.mu),
            "sigma2": float(self.sigma2),
            "n": int(self.n),
            "updated_unix_s": float(self.updated_unix_s),
            "flagged": bool(self.flagged),
            "detail": dict(self.detail),
        }


# ----------------------------- Frame Packet -------------------------------------------

@dataclass
class FramePacket:
    image: np.ndarray
    timestamp_unix_s: float
    config: ConfigKey
    pose_id: str
    phi_id: str
    intrinsics_id: str
    radiometric_id: str
    extra: Dict[str, Any]

    def meta(self) -> Dict[str, Any]:
        return {
            "timestamp_unix_s": self.timestamp_unix_s,
            "config": self.config.to_dict(),
            "pose_id": self.pose_id,
            "phi_id": self.phi_id,
            "intrinsics_id": self.intrinsics_id,
            "radiometric_id": self.radiometric_id,
            "extra": self.extra,
        }


# ----------------------------- Hardware Interfaces ------------------------------------

class MotorController:
    async def move_to_index(self, i: int) -> None:
        raise NotImplementedError

    async def read_index(self) -> int:
        raise NotImplementedError

    async def is_settled(self) -> bool:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError


class CameraDevice:
    async def capture_frame(self, band: Band) -> np.ndarray:
        raise NotImplementedError


class CalibrationSource:
    async def set_enabled(self, enabled: bool) -> None:
        raise NotImplementedError

    async def set_temperature_K(self, T_K: float) -> None:
        raise NotImplementedError

    async def read_temperature_K(self) -> float:
        raise NotImplementedError


# ----------------------------- Simulation Hardware ------------------------------------

class SimMotor(MotorController):
    def __init__(self, settle_time_s: float = 0.15, jitter_s: float = 0.03):
        self._idx = 1
        self._target = 1
        self._moving_until = 0.0
        self._settle_time_s = settle_time_s
        self._jitter_s = jitter_s

    async def move_to_index(self, i: int) -> None:
        self._target = i
        now = time.time()
        travel = self._settle_time_s + random.random() * self._jitter_s
        self._moving_until = now + travel
        await asyncio.sleep(travel)
        self._idx = i

    async def read_index(self) -> int:
        return self._idx

    async def is_settled(self) -> bool:
        return time.time() >= self._moving_until

    async def stop(self) -> None:
        self._moving_until = time.time()
        self._target = self._idx


class SimCalibrationSource(CalibrationSource):
    def __init__(self):
        self._enabled = False
        self._T = 300.0

    async def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    async def set_temperature_K(self, T_K: float) -> None:
        self._T = float(T_K)

    async def read_temperature_K(self) -> float:
        return self._T + np.random.randn() * 0.05

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def T(self) -> float:
        return self._T


class SimCamera(CameraDevice):
    """
    Simulation camera outputs a base scene plus a calibration ROI patch when cal source enabled.
    We also inject mild nonlinearity & band dependence to stress the fit.
    """
    def __init__(self, intrinsics_by_band: Dict[Band, BandIntrinsics], cal: SimCalibrationSource):
        self._intr = intrinsics_by_band
        self._cal = cal

    async def capture_frame(self, band: Band) -> np.ndarray:
        w, h = self._intr[band].resolution
        yy, xx = np.mgrid[0:h, 0:w]
        base = 0.15 + 0.25 * (xx / max(w - 1, 1)) + 0.10 * (yy / max(h - 1, 1))
        img = base + np.random.randn(h, w) * 0.01

        if self._cal.enabled:
            T = self._cal.T
            # "true" radiance proxy per band
            L = self._toy_Lcal(band, T)
            # Mild nonlinearity in observed DN: y = a*L + c + q*L^2
            if band == "b1":
                a, c, q = 1.05, 0.01, 0.12
            else:
                a, c, q = 0.95, 0.015, 0.08
            y_cal = a * L + c + q * (L ** 2)
            patch = y_cal + np.random.randn(70, 110) * 0.002
            img[10:80, 10:120] = patch

        return img.astype(np.float32)

    @staticmethod
    def _toy_Lcal(band: Band, T: float) -> float:
        scale = 0.002 if band == "b1" else 0.0015
        return scale * (T - 250.0)


# ----------------------------- Commands and Controller --------------------------------

class Mode(enum.Enum):
    CAPTURE = "capture"
    CALIBRATE = "calibrate"


@dataclass
class Command:
    mode: Mode
    config: ConfigKey
    n_frames: int = 1

    # multi-point calibration schedule
    calibrate_temperatures_K: Optional[List[float]] = None  # if None, use default
    calibrate_frames_per_point: int = 5

    # fitting configuration
    fit_kind: RadiometricModelKind = RadiometricModelKind.LINEAR
    # for mild nonlinearity you can choose QUADRATIC; if auto=True we select by AIC
    fit_auto_select: bool = True

    # validation thresholds (per band override allowed)
    drift_mu_thresh: Optional[float] = None
    drift_sigma2_thresh: Optional[float] = None
    max_fit_rms: Optional[float] = None  # fail calibration if RMS too large

    note: str = ""


class ControllerState(enum.Enum):
    IDLE = "IDLE"
    SWITCHING = "SWITCHING"
    SETTLING = "SETTLING"
    CAPTURING = "CAPTURING"
    CALIBRATING = "CALIBRATING"
    VALIDATING = "VALIDATING"
    ERROR = "ERROR"


@dataclass
class ConfigDB:
    poses_by_i: Dict[int, PoseSE3]
    intrinsics_by_band: Dict[Band, BandIntrinsics]
    phi_by_path_band: Dict[Tuple[Path, Band], PhiModel]

    # per κ radiometry and drift
    radiometric_fit_by_kappa: Dict[ConfigKey, RadiometricFit]
    drift_by_kappa: Dict[ConfigKey, DriftStats]

    # calibration ROI per κ (optional); if absent use default per band
    cal_roi_by_kappa: Dict[ConfigKey, Tuple[slice, slice]]

    def pose_id(self, i: int) -> str:
        return f"pose:i={i}"

    def intrinsics_id(self, b: Band) -> str:
        return f"intr:{b}"

    def phi_id(self, p: Path, b: Band) -> str:
        return self.phi_by_path_band[(p, b)].id()

    def radiometric_id(self, k: ConfigKey) -> str:
        fit = self.radiometric_fit_by_kappa.get(k)
        if fit is None:
            return "rad:unset"
        # deterministic-ish id
        return f"rad:{fit.kind}:rms={fit.rms:.4g}:t={int(fit.updated_unix_s)}"


class OrthospaceController:
    def __init__(
        self,
        db: ConfigDB,
        motor: MotorController,
        camera: CameraDevice,
        cal: CalibrationSource,
    ):
        self.db = db
        self.motor = motor
        self.camera = camera
        self.cal = cal
        self.state = ControllerState.IDLE
        self._q: asyncio.Queue[Command] = asyncio.Queue()
        self._last_error: Optional[str] = None

        # default drift thresholds
        self.default_drift_mu_thresh = 0.02
        self.default_drift_sigma2_thresh = 0.0025
        self.default_max_fit_rms = 0.01

    async def enqueue(self, cmd: Command) -> None:
        await self._q.put(cmd)

    async def run_forever(self) -> None:
        while True:
            cmd = await self._q.get()
            try:
                await self._execute(cmd)
            except Exception as e:
                self.state = ControllerState.ERROR
                self._last_error = f"{type(e).__name__}: {e}"
                await self._failsafe()
            finally:
                self._q.task_done()

    async def _failsafe(self) -> None:
        try:
            await self.motor.stop()
        except Exception:
            pass
        try:
            await self.cal.set_enabled(False)
        except Exception:
            pass

    async def _execute(self, cmd: Command) -> None:
        if cmd.mode == Mode.CAPTURE:
            await self._capture_sequence(cmd)
        elif cmd.mode == Mode.CALIBRATE:
            await self._calibrate_sequence(cmd)
        else:
            raise ValueError(f"Unknown mode: {cmd.mode}")

    async def _move_and_settle(self, i: int) -> None:
        self.state = ControllerState.SWITCHING
        await self.motor.move_to_index(i)

        self.state = ControllerState.SETTLING
        idx = await self.motor.read_index()
        if idx != i:
            raise RuntimeError(f"Motor index mismatch: commanded={i}, read={idx}")

        t0 = time.time()
        while True:
            if await self.motor.is_settled():
                break
            if time.time() - t0 > 2.0:
                raise RuntimeError("Motor settle timeout")
            await asyncio.sleep(0.01)

    async def _capture_sequence(self, cmd: Command) -> List[FramePacket]:
        k = cmd.config
        await self._move_and_settle(k.i)
        self.state = ControllerState.CAPTURING
        await self.cal.set_enabled(False)

        out: List[FramePacket] = []
        for _ in range(max(1, cmd.n_frames)):
            img = await self.camera.capture_frame(k.b)
            pkt = FramePacket(
                image=img,
                timestamp_unix_s=time.time(),
                config=k,
                pose_id=self.db.pose_id(k.i),
                phi_id=self.db.phi_id(k.p, k.b),
                intrinsics_id=self.db.intrinsics_id(k.b),
                radiometric_id=self.db.radiometric_id(k),
                extra={"note": cmd.note, "mode": "capture"},
            )
            out.append(pkt)

        self._log_frames(out)
        self.state = ControllerState.IDLE
        return out

    async def _calibrate_sequence(self, cmd: Command) -> None:
        k = cmd.config
        await self._move_and_settle(k.i)
        self.state = ControllerState.CALIBRATING

        temps = cmd.calibrate_temperatures_K
        if not temps:
            # default 2-point schedule
            temps = [300.0, 340.0]

        # acquire (T, L, y) points (y = mean ROI)
        roi = self.db.cal_roi_by_kappa.get(k) or self._default_cal_roi(k.b)

        T_list: List[float] = []
        L_list: List[float] = []
        y_list: List[float] = []
        y_var_list: List[float] = []

        for T in temps:
            await self.cal.set_temperature_K(float(T))
            # Optional: in real HW, wait for thermal stabilization / tolerance window
            await asyncio.sleep(0.05)
            await self.cal.set_enabled(True)

            frames = []
            for _ in range(max(3, int(cmd.calibrate_frames_per_point))):
                frames.append(await self.camera.capture_frame(k.b))

            await self.cal.set_enabled(False)

            # read actual temperature (optional)
            T_read = float(await self.cal.read_temperature_K())
            L = float(self._Lcal(k.b, T_read))  # replace with Planck-bandpass integration
            y_vals = np.array([np.mean(f[roi]) for f in frames], dtype=float)

            T_list.append(T_read)
            L_list.append(L)
            y_list.append(float(np.mean(y_vals)))
            y_var_list.append(float(np.var(y_vals)))

        # fit model
        fit = self._fit_radiometric_model(
            k=k,
            L=np.array(L_list, dtype=float),
            y=np.array(y_list, dtype=float),
            kind=cmd.fit_kind,
            auto_select=cmd.fit_auto_select,
            temperatures_K=T_list,
            note=cmd.note,
        )

        # optional fit quality gate
        max_fit_rms = cmd.max_fit_rms if cmd.max_fit_rms is not None else self.default_max_fit_rms
        if fit.rms > max_fit_rms:
            raise RuntimeError(f"Radiometric fit RMS too high for κ={k}: rms={fit.rms:.6g} > {max_fit_rms:.6g}")

        self.db.radiometric_fit_by_kappa[k] = fit

        # validate drift by re-evaluating residuals against fit at the calibration points
        self.state = ControllerState.VALIDATING
        mu, sigma2, n = self._residual_stats(L=np.array(L_list), y=np.array(y_list), fit=fit)

        mu_thr = cmd.drift_mu_thresh if cmd.drift_mu_thresh is not None else self.default_drift_mu_thresh
        s2_thr = cmd.drift_sigma2_thresh if cmd.drift_sigma2_thresh is not None else self.default_drift_sigma2_thresh

        flagged = (abs(mu) > mu_thr) or (sigma2 > s2_thr)
        drift = DriftStats(
            mu=float(mu),
            sigma2=float(sigma2),
            n=int(n),
            updated_unix_s=time.time(),
            flagged=bool(flagged),
            detail={
                "mu_thresh": float(mu_thr),
                "sigma2_thresh": float(s2_thr),
                "fit_kind": str(fit.kind),
                "fit_rms": float(fit.rms),
                "L_points": [float(v) for v in L_list],
                "y_points": [float(v) for v in y_list],
                "T_points_K": [float(v) for v in T_list],
                "y_var_points": [float(v) for v in y_var_list],
            },
        )
        self.db.drift_by_kappa[k] = drift

        self._log_calibration(k, fit, drift)
        self.state = ControllerState.IDLE

    # ----------------------------- Radiometry -----------------------------------------

    def _Lcal(self, band: Band, T_K: float) -> float:
        """
        Calibration radiance model L_b^{cal}(T). Replace with Planck integral over bandpass.
        For demo we use a monotone proxy.
        """
        scale = 0.002 if band == "b1" else 0.0015
        return scale * (T_K - 250.0)

    def _fit_radiometric_model(
        self,
        k: ConfigKey,
        L: np.ndarray,
        y: np.ndarray,
        kind: RadiometricModelKind,
        auto_select: bool,
        temperatures_K: List[float],
        note: str,
    ) -> RadiometricFit:
        """
        Fit y(L). Supports:
          - LINEAR: y = a*L + c
          - QUADRATIC: y = a*L^2 + b*L + c
        If auto_select=True, choose by AIC (prefers linear unless quadratic is justified).
        """
        if L.ndim != 1 or y.ndim != 1 or len(L) != len(y):
            raise ValueError("L and y must be 1D arrays of equal length")
        if len(L) < 2:
            raise ValueError("Need at least 2 points for linear fit")
        if kind == RadiometricModelKind.QUADRATIC and len(L) < 3:
            raise ValueError("Need at least 3 points for quadratic fit")

        # Build candidate fits
        candidates: List[RadiometricFit] = []

        lin = self._fit_linear(L, y, temperatures_K, note)
        candidates.append(lin)

        if len(L) >= 3:
            quad = self._fit_quadratic(L, y, temperatures_K, note)
            candidates.append(quad)

        if not auto_select:
            # force requested kind when possible
            if kind == RadiometricModelKind.LINEAR:
                return lin
            if kind == RadiometricModelKind.QUADRATIC:
                if len(L) < 3:
                    return lin
                return quad

        # Auto-select by AIC
        best = min(candidates, key=lambda f: self._aic(L, y, f))
        return best

    def _fit_linear(self, L: np.ndarray, y: np.ndarray, temperatures_K: List[float], note: str) -> RadiometricFit:
        # y = a*L + c
        X = np.stack([L, np.ones_like(L)], axis=1)  # (n,2)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coeffs
        rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return RadiometricFit(
            kind=RadiometricModelKind.LINEAR,
            coeffs=coeffs.astype(float),
            L_domain=(float(np.min(L)), float(np.max(L))),
            rms=rms,
            updated_unix_s=time.time(),
            n_points=int(len(L)),
            temperatures_K=[float(t) for t in temperatures_K],
            notes={"note": note, "model": "y=a*L+c"},
        )

    def _fit_quadratic(self, L: np.ndarray, y: np.ndarray, temperatures_K: List[float], note: str) -> RadiometricFit:
        # y = a*L^2 + b*L + c
        X = np.stack([L**2, L, np.ones_like(L)], axis=1)  # (n,3)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coeffs
        rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return RadiometricFit(
            kind=RadiometricModelKind.QUADRATIC,
            coeffs=coeffs.astype(float),
            L_domain=(float(np.min(L)), float(np.max(L))),
            rms=rms,
            updated_unix_s=time.time(),
            n_points=int(len(L)),
            temperatures_K=[float(t) for t in temperatures_K],
            notes={"note": note, "model": "y=a*L^2+b*L+c"},
        )

    def _aic(self, L: np.ndarray, y: np.ndarray, fit: RadiometricFit) -> float:
        """
        Akaike Information Criterion for least squares with Gaussian noise:
          AIC = n*ln(RSS/n) + 2k
        where k = number of parameters.
        """
        yhat = fit.predict(L)
        rss = float(np.sum((y - yhat) ** 2))
        n = len(y)
        k = len(fit.coeffs)
        # numeric guard for perfect fit
        eps = 1e-12
        return n * math.log(max(rss / max(n, 1), eps)) + 2.0 * k

    def _residual_stats(self, L: np.ndarray, y: np.ndarray, fit: RadiometricFit) -> Tuple[float, float, int]:
        r = y - fit.predict(L)
        return float(np.mean(r)), float(np.var(r)), int(len(r))

    # ----------------------------- Calibration ROI ------------------------------------

    def _default_cal_roi(self, band: Band) -> Tuple[slice, slice]:
        # matches SimCamera cal patch location
        return (slice(10, 80), slice(10, 120))

    # ----------------------------- Logging --------------------------------------------

    def _log_frames(self, packets: List[FramePacket]) -> None:
        for pkt in packets:
            print("[FRAME]", json.dumps(pkt.meta(), separators=(",", ":"), sort_keys=True))

    def _log_calibration(self, k: ConfigKey, fit: RadiometricFit, drift: DriftStats) -> None:
        payload = {
            "kappa": k.to_dict(),
            "radiometric_fit": fit.to_dict(),
            "drift": drift.to_dict(),
        }
        print("[CAL]", json.dumps(payload, separators=(",", ":"), sort_keys=True))


# ----------------------------- Demo DB ------------------------------------------------

def _random_rotation_about_z(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def build_demo_db() -> ConfigDB:
    I = [1, 2, 3, 4]
    poses_by_i: Dict[int, PoseSE3] = {}
    for i, ang in zip(I, [0.0, 90.0, 180.0, 270.0]):
        poses_by_i[i] = PoseSE3(R=_random_rotation_about_z(ang), t=np.zeros(3, dtype=float))

    intrinsics_by_band: Dict[Band, BandIntrinsics] = {
        "b1": BandIntrinsics(
            K=np.array([[520.0, 0.0, 320.0],
                        [0.0, 520.0, 240.0],
                        [0.0,   0.0,   1.0]], dtype=float),
            dist=np.zeros(5, dtype=float),
            resolution=(640, 480),
        ),
        "b2": BandIntrinsics(
            K=np.array([[500.0, 0.0, 320.0],
                        [0.0, 500.0, 240.0],
                        [0.0,   0.0,   1.0]], dtype=float),
            dist=np.zeros(5, dtype=float),
            resolution=(640, 480),
        ),
    }

    phi_by_path_band: Dict[Tuple[Path, Band], PhiModel] = {}
    for b in ["b1", "b2"]:
        axis_by_i_p1: Dict[int, np.ndarray] = {}
        axis_by_i_p2: Dict[int, np.ndarray] = {}
        for i in I:
            ang = (i - 1) * 90.0
            R = _random_rotation_about_z(ang)
            a1 = R @ np.array([1.0, 0.0, 0.2], dtype=float)
            a2 = R @ np.array([0.0, 1.0, 0.2], dtype=float)
            a1 /= (np.linalg.norm(a1) + 1e-12)
            a2 /= (np.linalg.norm(a2) + 1e-12)
            axis_by_i_p1[i] = a1
            axis_by_i_p2[i] = a2

        phi_by_path_band[("p1", b)] = FirstOrderAfocalPhi(phi_name=f"p1_{b}", magnification=2.0 if b == "b1" else 2.2, axis_by_i=axis_by_i_p1)
        phi_by_path_band[("p2", b)] = FirstOrderAfocalPhi(phi_name=f"p2_{b}", magnification=1.5 if b == "b1" else 1.6, axis_by_i=axis_by_i_p2)

    return ConfigDB(
        poses_by_i=poses_by_i,
        intrinsics_by_band=intrinsics_by_band,
        phi_by_path_band=phi_by_path_band,
        radiometric_fit_by_kappa={},
        drift_by_kappa={},
        cal_roi_by_kappa={},
    )


# ----------------------------- Demo ---------------------------------------------------

async def demo() -> None:
    db = build_demo_db()
    cal = SimCalibrationSource()
    motor = SimMotor()
    cam = SimCamera(db.intrinsics_by_band, cal)
    ctrl = OrthospaceController(db, motor, cam, cal)

    task = asyncio.create_task(ctrl.run_forever())

    # Multi-point calibration per κ:
    # - auto-select between linear and quadratic by AIC
    # - use 3 points to allow quadratic if needed
    temps = [290.0, 320.0, 350.0]

    for p in ["p1", "p2"]:
        for b in ["b1", "b2"]:
            k = ConfigKey(i=1, p=p, b=b)
            await ctrl.enqueue(Command(
                mode=Mode.CALIBRATE,
                config=k,
                calibrate_temperatures_K=temps,
                calibrate_frames_per_point=6,
                fit_kind=RadiometricModelKind.QUADRATIC,  # desired
                fit_auto_select=True,                    # but choose best by AIC
                max_fit_rms=0.01,
                note="multipt_cal",
            ))

    # Capture across indices to show deterministic κ metadata
    for i in [1, 2, 3, 4]:
        for p in ["p1", "p2"]:
            for b in ["b1", "b2"]:
                await ctrl.enqueue(Command(
                    mode=Mode.CAPTURE,
                    config=ConfigKey(i=i, p=p, b=b),
                    n_frames=2,
                    note="survey",
                ))

    await ctrl._q.join()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="demo", choices=["demo"])
    args = ap.parse_args()
    asyncio.run(demo())


if __name__ == "__main__":
    main()
