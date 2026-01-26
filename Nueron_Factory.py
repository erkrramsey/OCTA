#!/usr/bin/env python3
"""
neuron_factory_full_stack_v7.py

NEURON FACTORY v7 — All-in-one:

- Multi-type meshes (LIF, IZH, GLIF, PYR_L5, INT_FS, INT_CH, INT_MR, CBL_PK, THAL_G, OMNI)
- NeuronFactory: multi-mesh wiring, stepping, summary
- Mesh utilities: ISI, correlations, spectral proxy, JSON IO
- Unsupervised mesh training (STDP + homeostasis) per neuron type, with PRESETS
- PyTorch OmniNeuronLayer + temporal supervised training loop, with save/load
- Task-specific mesh training:
    * Pattern classification (rate-coded + linear readout) per type, with PRESETS
    * Sequence copy / prediction (temporal decoder) per type, with PRESETS
- RL:
    * NeuronMeshEnv: target firing-rate control (with presets)
    * Actor-critic-style training loop, with save/load
- Serialization:
    * Mesh save/load (JSON/NPZ)
    * Classifier save/load (mesh + W_out)
    * Sequence decoder save/load (mesh + W_dec)
    * Omni Torch save/load (state_dicts + config)
    * RL Actor-Critic save/load (state_dict + config)
- Hyperparameter PRESETS per neuron type (+ API to enumerate & apply)
- FastAPI + HTML UI:
    * /                         -> control panel
    * /api/neuron_types         -> list neuron types
    * /api/tasks                -> list tasks
    * /api/presets              -> list presets per neuron type
    * /api/start_run            -> start background run (task, neuron_type, preset_name, save_prefix)
    * /api/run_status/{id}      -> poll run status/logs/last stats
    * /api/plot/{id}/raster     -> PNG raster plot for run mesh
    * /api/plot/{id}/rate       -> PNG rate plot for run mesh

"""

from __future__ import annotations

import argparse
import json
import os
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np

# Optional matplotlib (only needed if you call plotting helpers or API plots)
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Optional Gym (for RL-style API)
try:
    import gym
    from gym import spaces
    _HAS_GYM = True
except Exception:
    _HAS_GYM = False

# Optional FastAPI
try:
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    _HAS_FASTAPI = True
except Exception:
    _HAS_FASTAPI = False


# ----------------------------------------------------------------------
# Core enums and dataclasses
# ----------------------------------------------------------------------
class NeuronType(Enum):
    PYR_L5 = auto()       # Pyramidal L5 integrator / predictor
    INT_FS = auto()       # Fast-spiking basket interneuron
    INT_CH = auto()       # Chandelier veto cell
    INT_MR = auto()       # Martinotti apical inhibitor
    CBL_PK = auto()       # Purkinje-like teacher
    THAL_G = auto()       # Thalamic gatekeeper
    LIF = auto()          # Canonical LIF neuron
    IZH = auto()          # Izhikevich generic
    GLIF = auto()         # GLIF-style extended LIF
    OMNI = auto()         # Full omni-neuron


@dataclass
class MeshStats:
    neuron_type: str
    N: int
    mean_rate_hz: float
    peak_rate_hz: float
    mean_v: float
    frac_active: float


@dataclass
class NeuronMesh:
    """
    A homogeneous population of neurons of a given type, wired as a recurrent mesh.
    """
    neuron_type: NeuronType
    N: int
    dt: float = 1e-3  # seconds

    # Core state variables
    V: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)
    h_b: np.ndarray = field(init=False)
    h_a: np.ndarray = field(init=False)
    spikes: np.ndarray = field(init=False)

    # Connectivity
    W: np.ndarray = field(init=False)

    # Parameters
    params: Dict[str, float] = field(default_factory=dict)

    # Plasticity flags
    stdp_enabled: bool = False
    homeostasis_enabled: bool = False

    # Internal buffers
    time: float = 0.0
    spike_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.V = np.zeros(self.N, dtype=np.float32)
        self.U = np.zeros(self.N, dtype=np.float32)
        self.A = np.zeros(self.N, dtype=np.float32)
        self.h_b = np.zeros(self.N, dtype=np.float32)
        self.h_a = np.zeros(self.N, dtype=np.float32)
        self.spikes = np.zeros(self.N, dtype=np.float32)

        self.W = np.random.normal(loc=0.0, scale=0.05, size=(self.N, self.N)).astype(np.float32)
        np.fill_diagonal(self.W, 0.0)

        self._init_default_params()
        self.reset_state()

    # ------------------------------------------------------------------
    # Parameter initialization per neuron type
    # ------------------------------------------------------------------
    def _init_default_params(self):
        p = self.params
        t = self.neuron_type

        if t == NeuronType.LIF:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("V_th", -50.0)
            p.setdefault("tau_m", 20e-3)
            p.setdefault("R_m", 100e6)
        elif t == NeuronType.IZH:
            p.setdefault("a", 0.02)
            p.setdefault("b", 0.2)
            p.setdefault("c", -65.0)
            p.setdefault("d", 8.0)
        elif t == NeuronType.GLIF:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("V_th", -50.0)
            p.setdefault("tau_m", 20e-3)
            p.setdefault("R_m", 100e6)
            p.setdefault("tau_asc", 100e-3)
            p.setdefault("asc_ampl", 2.0)
            p.setdefault("theta_base", -50.0)
            p.setdefault("theta_slope", 2.0)
        elif t == NeuronType.PYR_L5:
            p.setdefault("V_rest", -70.0)
            p.setdefault("V_reset", -72.0)
            p.setdefault("V_th", -52.0)
            p.setdefault("tau_m", 30e-3)
            p.setdefault("R_m", 150e6)
            p.setdefault("tau_adapt", 200e-3)
            p.setdefault("adapt_gain", 0.5)
        elif t == NeuronType.INT_FS:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -67.0)
            p.setdefault("V_th", -50.0)
            p.setdefault("tau_m", 10e-3)
            p.setdefault("R_m", 80e6)
        elif t == NeuronType.INT_CH:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("V_th", -48.0)
            p.setdefault("tau_m", 15e-3)
            p.setdefault("R_m", 100e6)
            p.setdefault("veto_strength", 5.0)
        elif t == NeuronType.INT_MR:
            p.setdefault("V_rest", -68.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("V_th", -52.0)
            p.setdefault("tau_m", 25e-3)
            p.setdefault("R_m", 120e6)
        elif t == NeuronType.CBL_PK:
            p.setdefault("V_rest", -62.0)
            p.setdefault("V_reset", -65.0)
            p.setdefault("V_th", -45.0)
            p.setdefault("tau_m", 40e-3)
            p.setdefault("R_m", 100e6)
            p.setdefault("burst_th", -35.0)
            p.setdefault("burst_k", 3.0)
        elif t == NeuronType.THAL_G:
            p.setdefault("V_rest", -70.0)
            p.setdefault("V_reset", -72.0)
            p.setdefault("V_th_tonic", -50.0)
            p.setdefault("V_th_burst", -60.0)
            p.setdefault("tau_m", 20e-3)
            p.setdefault("R_m", 120e6)
            p.setdefault("burst_gain", 2.0)
        elif t == NeuronType.OMNI:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("theta_base", -50.0)
            p.setdefault("theta_burst", -40.0)
            p.setdefault("tau_m_base", 20e-3)
            p.setdefault("tau_m_min", 5e-3)
            p.setdefault("tau_m_max", 50e-3)
            p.setdefault("adapt_tau", 200e-3)
            p.setdefault("adapt_gain", 1.0)
            p.setdefault("burst_k", 3.0)
        else:
            p.setdefault("V_rest", -65.0)
            p.setdefault("V_reset", -70.0)
            p.setdefault("V_th", -50.0)
            p.setdefault("tau_m", 20e-3)
            p.setdefault("R_m", 100e6)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset_state(self):
        v0 = self.params.get("V_rest", -65.0)
        self.V.fill(v0)
        self.U.fill(0.0)
        self.A.fill(0.0)
        self.h_b.fill(0.0)
        self.h_a.fill(0.0)
        self.spikes.fill(0.0)
        self.time = 0.0
        self.spike_history.clear()

    def copy_state(self) -> Dict[str, np.ndarray]:
        return {
            "V": self.V.copy(),
            "U": self.U.copy(),
            "A": self.A.copy(),
            "h_b": self.h_b.copy(),
            "h_a": self.h_a.copy(),
            "spikes": self.spikes.copy(),
            "time": np.array(self.time),
        }

    def restore_state(self, state: Dict[str, np.ndarray]):
        self.V = state["V"].copy()
        self.U = state["U"].copy()
        self.A = state["A"].copy()
        self.h_b = state["h_b"].copy()
        self.h_a = state["h_a"].copy()
        self.spikes = state["spikes"].copy()
        self.time = float(state["time"])

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def randomize_connectivity(self, p_connect: float = 0.1, w_scale: float = 0.05, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        mask = rng.random((self.N, self.N)) < p_connect
        np.fill_diagonal(mask, 0)
        self.W = (rng.normal(loc=0.0, scale=w_scale, size=(self.N, self.N)) * mask).astype(np.float32)

    def scale_recurrent(self, factor: float):
        self.W *= factor

    def clamp_weights(self, w_min: float = -1.0, w_max: float = 1.0):
        np.clip(self.W, w_min, w_max, out=self.W)

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------
    def step(self, I_ext: np.ndarray, I_apical: Optional[np.ndarray] = None):
        if I_ext.shape != (self.N,):
            raise ValueError(f"I_ext shape {I_ext.shape} does not match N={self.N}")

        if I_apical is None:
            I_apical = np.zeros_like(I_ext)

        I_rec = self.W @ self.spikes
        I_basal = I_ext + I_rec

        t = self.neuron_type
        if t == NeuronType.LIF:
            self._step_lif(I_basal)
        elif t == NeuronType.IZH:
            self._step_izh(I_basal)
        elif t == NeuronType.GLIF:
            self._step_glif(I_basal)
        elif t == NeuronType.PYR_L5:
            self._step_pyr(I_basal, I_apical)
        elif t == NeuronType.INT_FS:
            self._step_fs(I_basal)
        elif t == NeuronType.INT_CH:
            self._step_chandelier(I_basal)
        elif t == NeuronType.INT_MR:
            self._step_martinotti(I_basal, I_apical)
        elif t == NeuronType.CBL_PK:
            self._step_purkinje(I_basal, I_apical)
        elif t == NeuronType.THAL_G:
            self._step_thalamic(I_basal)
        elif t == NeuronType.OMNI:
            self._step_omni(I_basal, I_apical)
        else:
            self._step_lif(I_basal)

        if self.stdp_enabled:
            self._apply_stdp(I_basal)

        if self.homeostasis_enabled:
            self._apply_homeostasis()

        self.spike_history.append(self.spikes.copy())
        self.time += self.dt

    # ------------------------------------------------------------------
    # Specific neuron model updates
    # ------------------------------------------------------------------
    def _step_lif(self, I_basal: np.ndarray):
        p = self.params
        V = self.V
        dt = self.dt

        dV = (-(V - p["V_rest"]) + p["R_m"] * I_basal) / p["tau_m"]
        V_new = V + dt * dV

        spk = (V_new >= p["V_th"]).astype(np.float32)
        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_izh(self, I_basal: np.ndarray):
        p = self.params
        V = self.V
        U = self.U
        dt = self.dt

        dv = 0.04 * V * V + 5 * V + 140 - U + I_basal
        du = p["a"] * (p["b"] * V - U)

        V_new = V + dt * dv
        U_new = U + dt * du

        spk = (V_new >= 30.0).astype(np.float32)
        V_new = np.where(spk > 0, p["c"], V_new)
        U_new = np.where(spk > 0, U_new + p["d"], U_new)

        self.V = V_new
        self.U = U_new
        self.spikes = spk

    def _step_glif(self, I_basal: np.ndarray):
        p = self.params
        V = self.V
        A = self.A
        dt = self.dt

        dA = -A / p["tau_asc"]
        A_new = A + dt * dA

        theta = p["theta_base"] + p["theta_slope"] * A_new

        dV = (-(V - p["V_rest"]) + p["R_m"] * I_basal - A_new) / p["tau_m"]
        V_new = V + dt * dV

        spk = (V_new >= theta).astype(np.float32)
        V_new = np.where(spk > 0, p["V_reset"], V_new)
        A_new = np.where(spk > 0, A_new + p["asc_ampl"], A_new)

        self.V = V_new
        self.A = A_new
        self.spikes = spk

    def _step_pyr(self, I_basal: np.ndarray, I_apical: np.ndarray):
        p = self.params
        dt = self.dt

        tau_m = p["tau_m"]
        self.h_b += dt * (-(self.h_b) / tau_m + I_basal)
        self.h_a += dt * (-(self.h_a) / tau_m + I_apical)

        drive = self.h_b + 0.7 * self.h_a

        self.A += dt * (-self.A / p["tau_adapt"])
        eff_th = p["V_th"] + p["adapt_gain"] * self.A

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * drive) / p["tau_m"]
        V_new = self.V + dt * dV

        spk = (V_new >= eff_th).astype(np.float32)
        V_new = np.where(spk > 0, p["V_reset"], V_new)
        self.A = np.where(spk > 0, self.A + 1.0, self.A)

        self.V = V_new
        self.spikes = spk

    def _step_fs(self, I_basal: np.ndarray):
        p = self.params
        dt = self.dt

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * I_basal) / p["tau_m"]
        V_new = self.V + dt * dV

        spk = (V_new >= p["V_th"]).astype(np.float32)
        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_chandelier(self, I_basal: np.ndarray):
        p = self.params
        dt = self.dt

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * I_basal) / p["tau_m"]
        V_new = self.V + dt * dV

        readiness = 1.0 / (1.0 + np.exp(-(V_new - p["V_th"]) / 2.0))
        spk = (readiness * p["veto_strength"] > 1.0).astype(np.float32)

        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_martinotti(self, I_basal: np.ndarray, I_apical: np.ndarray):
        p = self.params
        dt = self.dt

        drive = I_basal + 1.5 * I_apical

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * drive) / p["tau_m"]
        V_new = self.V + dt * dV

        spk = (V_new >= p["V_th"]).astype(np.float32)
        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_purkinje(self, I_basal: np.ndarray, I_apical: np.ndarray):
        p = self.params
        dt = self.dt

        drive = I_basal + 0.3 * np.random.normal(0.0, 1.0, size=self.N)

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * drive) / p["tau_m"]
        V_new = self.V + dt * dV

        burst_trigger = I_apical > 0.5 * np.max(np.abs(I_apical) + 1e-6)

        simple_spk = (V_new >= p["V_th"]).astype(np.float32)
        burst_spk = burst_trigger.astype(np.float32) * p["burst_k"]

        spk = np.maximum(simple_spk, burst_spk)
        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_thalamic(self, I_basal: np.ndarray):
        p = self.params
        dt = self.dt

        below_rest = np.minimum(0.0, self.V - p["V_rest"])
        self.A += dt * (-self.A / 0.1 + below_rest)

        tonic_th = p["V_th_tonic"]
        burst_th = p["V_th_burst"]

        dV = (-(self.V - p["V_rest"]) + p["R_m"] * I_basal) / p["tau_m"]
        V_new = self.V + dt * dV

        burst_mode = self.A < -5.0

        tonic_spk = (V_new >= tonic_th).astype(np.float32)
        burst_spk = ((V_new >= burst_th) & burst_mode).astype(np.float32) * p["burst_gain"]

        spk = np.maximum(tonic_spk, burst_spk)
        V_new = np.where(spk > 0, p["V_reset"], V_new)

        self.V = V_new
        self.spikes = spk

    def _step_omni(self, I_basal: np.ndarray, I_apical: np.ndarray):
        p = self.params
        dt = self.dt

        tau_m_base = p["tau_m_base"]
        self.h_b += dt * (-(self.h_b) / tau_m_base + I_basal)
        self.h_a += dt * (-(self.h_a) / tau_m_base + I_apical)

        g_eff = np.abs(self.h_b) + 0.5 * np.abs(self.h_a)
        tau_dynamic = p["tau_m_min"] + (p["tau_m_max"] - p["tau_m_min"]) * np.exp(-g_eff)
        tau_dynamic = np.clip(tau_dynamic, p["tau_m_min"], p["tau_m_max"])

        self.A += dt * (-self.A / p["adapt_tau"])

        theta = p["theta_base"] + p["adapt_gain"] * self.A

        drive = self.h_b + 0.7 * self.h_a
        dV = (-(self.V - p["V_rest"]) + drive) / tau_dynamic
        V_new = self.V + dt * dV

        spk = np.zeros_like(self.V)
        burst_mask = V_new >= p["theta_burst"]
        spike_mask = (V_new >= theta) & ~burst_mask

        spk[spike_mask] = 1.0
        spk[burst_mask] = p["burst_k"]

        V_new = np.where(spk > 0, p["V_reset"], V_new)
        self.A = np.where(spk > 0, self.A + spk, self.A)

        self.V = V_new
        self.spikes = spk

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------
    def enable_stdp(self, enabled: bool = True):
        self.stdp_enabled = enabled

    def enable_homeostasis(self, enabled: bool = True):
        self.homeostasis_enabled = enabled

    def _apply_stdp(self, I_basal: np.ndarray, eta: float = 1e-3, tau_pre: float = 20e-3, tau_post: float = 20e-3):
        dt = self.dt

        self.U += dt * (-self.U / tau_pre + self.spikes)
        self.A += dt * (-self.A / tau_post + self.spikes)

        dW = eta * (np.outer(self.spikes, self.U) - np.outer(self.A, self.spikes))
        self.W += dW.astype(np.float32)

        self.clamp_weights(-1.0, 1.0)

    def _apply_homeostasis(self, target_rate_hz: float = 5.0, tau_rate: float = 5.0):
        dt = self.dt
        inst_rate = self.spikes / dt
        self.U += dt * (-(self.U) / tau_rate + inst_rate)

        mean_rate = np.mean(self.U)
        if mean_rate > 0:
            scale = target_rate_hz / (mean_rate + 1e-6)
            self.W *= scale ** 0.01

    # ------------------------------------------------------------------
    # Utilities and analysis
    # ------------------------------------------------------------------
    def run(self, T: float, input_fn: Optional[Callable[[float, int], Tuple[np.ndarray, np.ndarray]]] = None):
        steps = int(T / self.dt)
        for _ in range(steps):
            t = self.time
            if input_fn is None:
                I_basal = np.zeros(self.N, dtype=np.float32)
                I_apical = np.zeros(self.N, dtype=np.float32)
            else:
                I_basal, I_apical = input_fn(t, self.N)
            self.step(I_basal, I_apical)

    def get_spike_matrix(self) -> np.ndarray:
        if not self.spike_history:
            return np.zeros((0, self.N), dtype=np.float32)
        return np.stack(self.spike_history, axis=0)

    def compute_stats(self) -> MeshStats:
        S = self.get_spike_matrix()
        if S.size == 0:
            mean_rate = 0.0
            peak_rate = 0.0
            frac_active = 0.0
        else:
            inst_rate = S / self.dt
            mean_rate = float(np.mean(inst_rate))
            peak_rate = float(np.max(inst_rate))
            frac_active = float(np.mean(S > 0))

        return MeshStats(
            neuron_type=self.neuron_type.name,
            N=self.N,
            mean_rate_hz=mean_rate,
            peak_rate_hz=peak_rate,
            mean_v=float(np.mean(self.V)),
            frac_active=frac_active,
        )

    def estimate_isi(self, neuron_idx: int) -> np.ndarray:
        S = self.get_spike_matrix()
        if S.size == 0 or neuron_idx < 0 or neuron_idx >= self.N:
            return np.array([])
        spk_times = np.where(S[:, neuron_idx] > 0)[0] * self.dt
        if spk_times.size < 2:
            return np.array([])
        return np.diff(spk_times)

    def spike_correlation_matrix(self) -> np.ndarray:
        S = self.get_spike_matrix()
        if S.size == 0:
            return np.zeros((self.N, self.N), dtype=np.float32)
        X = S - np.mean(S, axis=0, keepdims=True)
        cov = X.T @ X
        std = np.sqrt(np.diag(cov) + 1e-8)
        denom = np.outer(std, std) + 1e-8
        return (cov / denom).astype(np.float32)

    def spectral_proxy(self) -> Tuple[np.ndarray, np.ndarray]:
        S = self.get_spike_matrix()
        if S.size == 0:
            return np.array([]), np.array([])
        T, N = S.shape
        X = S - np.mean(S, axis=0, keepdims=True)
        F = np.fft.rfft(X, axis=0)
        P = np.mean(np.abs(F) ** 2, axis=1)
        freqs = np.fft.rfftfreq(T, d=self.dt)
        return freqs, P

    def to_json(self) -> str:
        data = {
            "neuron_type": self.neuron_type.name,
            "N": self.N,
            "dt": self.dt,
            "params": self.params,
            "W": self.W.tolist(),
        }
        return json.dumps(data)

    @staticmethod
    def from_json(s: str) -> "NeuronMesh":
        data = json.loads(s)
        mesh = NeuronMesh(neuron_type=NeuronType[data["neuron_type"]], N=data["N"], dt=data["dt"], params=data["params"])
        mesh.W = np.array(data["W"], dtype=np.float32)
        mesh.reset_state()
        return mesh


# ----------------------------------------------------------------------
# Neuron Factory
# ----------------------------------------------------------------------
@dataclass
class NeuronFactory:
    meshes: Dict[str, NeuronMesh] = field(default_factory=dict)
    connections: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    def add_mesh(self, name: str, neuron_type: NeuronType, N: int, dt: float = 1e-3, **params) -> NeuronMesh:
        mesh = NeuronMesh(neuron_type=neuron_type, N=N, dt=dt, params=params)
        self.meshes[name] = mesh
        return mesh

    def connect_meshes(self, src: str, dst: str, p_connect: float = 0.2, w_scale: float = 0.05, seed: Optional[int] = None):
        if src not in self.meshes or dst not in self.meshes:
            raise KeyError("Both src and dst meshes must exist")

        src_mesh = self.meshes[src]
        dst_mesh = self.meshes[dst]

        rng = np.random.default_rng(seed)
        W = rng.normal(loc=0.0, scale=w_scale, size=(dst_mesh.N, src_mesh.N)).astype(np.float32)
        mask = rng.random((dst_mesh.N, src_mesh.N)) < p_connect
        W *= mask

        self.connections[(src, dst)] = W

    def step_all(self, input_map: Optional[Dict[str, Callable[[float, int], Tuple[np.ndarray, np.ndarray]]]] = None):
        mesh_inputs_basal: Dict[str, np.ndarray] = {}
        mesh_inputs_apical: Dict[str, np.ndarray] = {}

        for name, mesh in self.meshes.items():
            mesh_inputs_basal[name] = np.zeros(mesh.N, dtype=np.float32)
            mesh_inputs_apical[name] = np.zeros(mesh.N, dtype=np.float32)

        for (src, dst), W in self.connections.items():
            src_mesh = self.meshes[src]
            dst_mesh = self.meshes[dst]
            I_ff = W @ src_mesh.spikes
            mesh_inputs_basal[dst] += I_ff

        for name, mesh in self.meshes.items():
            t = mesh.time
            I_b_ext = mesh_inputs_basal[name]
            I_a_ext = mesh_inputs_apical[name]
            if input_map and name in input_map:
                I_b_add, I_a_add = input_map[name](t, mesh.N)
                I_b_ext = I_b_ext + I_b_add
                I_a_ext = I_a_ext + I_a_add
            mesh.step(I_b_ext, I_a_ext)

    def run_all(self, T: float, input_map: Optional[Dict[str, Callable[[float, int], Tuple[np.ndarray, np.ndarray]]]] = None):
        if not self.meshes:
            return
        dt = next(iter(self.meshes.values())).dt
        steps = int(T / dt)
        for _ in range(steps):
            self.step_all(input_map=input_map)

    def summary(self) -> Dict[str, MeshStats]:
        return {name: mesh.compute_stats() for name, mesh in self.meshes.items()}


# ----------------------------------------------------------------------
# HYPERPARAMETER PRESETS
# ----------------------------------------------------------------------
NEURON_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "OMNI": {
        "default": {},
        "fast_memory": {
            "tau_m_min": 0.005,
            "tau_m_max": 0.03,
            "adapt_tau": 0.1,
            "adapt_gain": 0.8,
        },
        "slow_memory": {
            "tau_m_min": 0.02,
            "tau_m_max": 0.1,
            "adapt_tau": 0.5,
            "adapt_gain": 0.6,
        },
        "burst_confidence": {
            "theta_burst": -45.0,
            "burst_k": 4.0,
            "adapt_gain": 1.2,
        },
    },
    "PYR_L5": {
        "default": {},
        "strong_predictive": {
            "tau_m": 0.04,
            "tau_adapt": 0.3,
            "adapt_gain": 0.8,
        },
        "weak_adapt": {
            "tau_adapt": 0.1,
            "adapt_gain": 0.3,
        },
    },
    "INT_FS": {
        "default": {},
        "ultra_fast": {
            "tau_m": 0.005,
            "R_m": 50e6,
        },
        "gamma_pacer": {
            "tau_m": 0.01,
            "V_th": -48.0,
        },
    },
    "INT_CH": {
        "default": {},
        "strong_veto": {
            "veto_strength": 7.0,
            "tau_m": 0.012,
        },
        "subtle_control": {
            "veto_strength": 3.0,
        },
    },
    "INT_MR": {
        "default": {},
        "strong_apical_control": {
            "tau_m": 0.03,
            "V_th": -54.0,
        },
    },
    "CBL_PK": {
        "default": {},
        "high_burst": {
            "burst_th": -40.0,
            "burst_k": 4.0,
        },
        "sensitive_teacher": {
            "V_th": -48.0,
            "tau_m": 0.05,
        },
    },
    "THAL_G": {
        "default": {},
        "light_sleep": {
            "V_th_burst": -58.0,
            "burst_gain": 1.5,
        },
        "deep_sleep": {
            "V_th_burst": -62.0,
            "burst_gain": 3.0,
        },
    },
    "LIF": {
        "default": {},
        "slow_integrator": {
            "tau_m": 0.05,
            "R_m": 150e6,
        },
        "fast_integrator": {
            "tau_m": 0.01,
            "R_m": 80e6,
        },
    },
    "IZH": {
        "default": {},
        "regular_spiking": {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
        },
        "chattering": {
            "a": 0.02,
            "b": 0.2,
            "c": -50.0,
            "d": 2.0,
        },
        "fast_spiking": {
            "a": 0.1,
            "b": 0.2,
            "c": -65.0,
            "d": 2.0,
        },
    },
    "GLIF": {
        "default": {},
        "strong_adapt": {
            "tau_asc": 0.15,
            "asc_ampl": 3.0,
            "theta_slope": 3.0,
        },
        "weak_adapt": {
            "tau_asc": 0.05,
            "asc_ampl": 1.0,
        },
    },
}


def apply_preset(mesh: NeuronMesh, preset_name: Optional[str]) -> None:
    if not preset_name:
        return
    ntype = mesh.neuron_type.name
    presets_for_type = NEURON_PRESETS.get(ntype, {})
    preset = presets_for_type.get(preset_name)
    if not preset:
        return
    mesh.params.update(preset)
    mesh.reset_state()


# ----------------------------------------------------------------------
# SERIALIZATION HELPERS
# ----------------------------------------------------------------------
def _ensure_dir_from_prefix(prefix: str):
    path = Path(prefix)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_mesh(mesh: NeuronMesh, prefix: str):
    _ensure_dir_from_prefix(prefix)
    mesh_json_path = f"{prefix}.mesh.json"
    with open(mesh_json_path, "w") as f:
        f.write(mesh.to_json())
    print(f"[SAVE] Mesh ({mesh.neuron_type.name}) -> {mesh_json_path}")


def load_mesh(prefix: str) -> NeuronMesh:
    mesh_json_path = f"{prefix}.mesh.json"
    with open(mesh_json_path, "r") as f:
        s = f.read()
    mesh = NeuronMesh.from_json(s)
    print(f"[LOAD] Mesh ({mesh.neuron_type.name}) <- {mesh_json_path}")
    return mesh


def save_classifier(mesh: NeuronMesh, W_out: np.ndarray, prefix: str):
    _ensure_dir_from_prefix(prefix)
    save_mesh(mesh, prefix)
    npz_path = f"{prefix}.classifier.npz"
    np.savez(npz_path, W_out=W_out.astype(np.float32))
    print(f"[SAVE] Classifier W_out -> {npz_path}")


def load_classifier(prefix: str) -> Tuple[NeuronMesh, np.ndarray]:
    mesh = load_mesh(prefix)
    npz_path = f"{prefix}.classifier.npz"
    data = np.load(npz_path)
    W_out = data["W_out"]
    print(f"[LOAD] Classifier W_out <- {npz_path}")
    return mesh, W_out


def save_seq_decoder(mesh: NeuronMesh, W_dec: np.ndarray, prefix: str):
    _ensure_dir_from_prefix(prefix)
    save_mesh(mesh, prefix)
    npz_path = f"{prefix}.seq.npz"
    np.savez(npz_path, W_dec=W_dec.astype(np.float32))
    print(f"[SAVE] Sequence decoder W_dec -> {npz_path}")


def load_seq_decoder(prefix: str) -> Tuple[NeuronMesh, np.ndarray]:
    mesh = load_mesh(prefix)
    npz_path = f"{prefix}.seq.npz"
    data = np.load(npz_path)
    W_dec = data["W_dec"]
    print(f"[LOAD] Sequence decoder W_dec <- {npz_path}")
    return mesh, W_dec


if _HAS_TORCH:
    def save_omni_torch(model, readout, prefix: str):
        _ensure_dir_from_prefix(prefix)
        pt_path = f"{prefix}.omni_torch.pt"
        cfg_path = f"{prefix}.omni_torch_config.json"

        state = {
            "model_state": model.state_dict(),
            "readout_state": readout.state_dict(),
        }
        torch.save(state, pt_path)

        cfg = {
            "in_features_basal": model.in_features_basal,
            "in_features_apical": model.in_features_apical,
            "hidden_size": model.hidden_size,
            "dt": float(model.dt),
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"[SAVE] Omni Torch -> {pt_path}, {cfg_path}")

    def load_omni_torch(prefix: str):
        pt_path = f"{prefix}.omni_torch.pt"
        cfg_path = f"{prefix}.omni_torch_config.json"

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        model = OmniNeuronLayer(
            in_features_basal=cfg["in_features_basal"],
            in_features_apical=cfg["in_features_apical"],
            hidden_size=cfg["hidden_size"],
            dt=cfg["dt"],
            use_hard_spikes=False,
        )
        readout = nn.Linear(cfg["hidden_size"], 1)

        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        readout.load_state_dict(state["readout_state"])

        print(f"[LOAD] Omni Torch <- {pt_path}, {cfg_path}")
        return model, readout


if _HAS_TORCH:
    def save_actor_critic(model: nn.Module, prefix: str):
        _ensure_dir_from_prefix(prefix)
        pt_path = f"{prefix}.rl.pt"
        cfg_path = f"{prefix}.rl_config.json"

        state = model.state_dict()
        torch.save(state, pt_path)

        cfg = {"obs_dim": model.fc.in_features, "hidden": model.fc.out_features}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"[SAVE] RL ActorCritic -> {pt_path}, {cfg_path}")

    def load_actor_critic(prefix: str):
        pt_path = f"{prefix}.rl.pt"
        cfg_path = f"{prefix}.rl_config.json"
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        class ActorCritic(nn.Module):
            def __init__(self, obs_dim: int, hidden: int = 64):
                super().__init__()
                self.fc = nn.Linear(obs_dim, hidden)
                self.policy = nn.Linear(hidden, 1)
                self.value = nn.Linear(hidden, 1)

            def forward(self, x):
                h = torch.tanh(self.fc(x))
                mean = torch.tanh(self.policy(h))
                value = self.value(h)
                return mean, value

        model = ActorCritic(obs_dim=cfg["obs_dim"], hidden=cfg["hidden"])
        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[LOAD] RL ActorCritic <- {pt_path}, {cfg_path}")
        return model


# ----------------------------------------------------------------------
# GLOBAL RUN REGISTRY (for API live status + mesh refs)
# ----------------------------------------------------------------------
RUN_REGISTRY: Dict[str, Dict[str, Any]] = {}
RUN_COUNTER = itertools.count()


def create_run(task: str, neuron_type: Optional[str], params: Dict[str, Any]) -> str:
    run_id = f"run_{next(RUN_COUNTER)}"
    RUN_REGISTRY[run_id] = {
        "task": task,
        "neuron_type": neuron_type,
        "params": params,
        "status": "pending",
        "progress": 0.0,
        "logs": [],
        "last_stats": None,
        "error": None,
        "mesh": None,     # will hold a NeuronMesh if applicable
    }
    return run_id


def log_run(run_id: str, msg: str, progress: Optional[float] = None, stats: Optional[MeshStats] = None):
    if run_id not in RUN_REGISTRY:
        return
    entry = RUN_REGISTRY[run_id]
    entry["logs"].append(msg)
    if progress is not None:
        entry["progress"] = float(progress)
    if stats is not None:
        entry["last_stats"] = {
            "neuron_type": stats.neuron_type,
            "N": stats.N,
            "mean_rate_hz": stats.mean_rate_hz,
            "peak_rate_hz": stats.peak_rate_hz,
            "mean_v": stats.mean_v,
            "frac_active": stats.frac_active,
        }


# ----------------------------------------------------------------------
# PyTorch Omni-Neuron Layer (and training)
# ----------------------------------------------------------------------
if _HAS_TORCH:

    class OmniNeuronLayer(nn.Module):
        def __init__(
            self,
            in_features_basal: int,
            in_features_apical: int,
            hidden_size: int,
            theta_base: float = -1.0,
            theta_burst: float = 1.0,
            tau_m_min: float = 0.01,
            tau_m_max: float = 0.1,
            adapt_tau: float = 0.2,
            adapt_gain: float = 0.5,
            burst_k: float = 2.0,
            dt: float = 1e-3,
            use_hard_spikes: bool = False,
        ):
            super().__init__()
            self.in_features_basal = in_features_basal
            self.in_features_apical = in_features_apical
            self.hidden_size = hidden_size
            self.dt = dt
            self.use_hard_spikes = use_hard_spikes

            self.W_b = nn.Linear(in_features_basal, hidden_size)
            self.W_a = nn.Linear(in_features_apical, hidden_size)

            self.theta_base = nn.Parameter(torch.tensor(theta_base))
            self.theta_burst = nn.Parameter(torch.tensor(theta_burst))
            self.tau_m_min = nn.Parameter(torch.tensor(tau_m_min))
            self.tau_m_max = nn.Parameter(torch.tensor(tau_m_max))
            self.adapt_tau = nn.Parameter(torch.tensor(adapt_tau))
            self.adapt_gain = nn.Parameter(torch.tensor(adapt_gain))
            self.burst_k = nn.Parameter(torch.tensor(burst_k))

        def init_state(self, batch_size: int, device=None) -> Dict[str, torch.Tensor]:
            if device is None:
                device = next(self.parameters()).device
            V = torch.zeros(batch_size, self.hidden_size, device=device)
            A = torch.zeros_like(V)
            h_b = torch.zeros_like(V)
            h_a = torch.zeros_like(V)
            return {"V": V, "A": A, "h_b": h_b, "h_a": h_a}

        def forward(
            self,
            x_b: torch.Tensor,
            x_a: torch.Tensor,
            state: Optional[Dict[str, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            if state is None:
                state = self.init_state(x_b.size(0), device=x_b.device)

            V = state["V"]
            A = state["A"]
            h_b = state["h_b"]
            h_a = state["h_a"]

            I_b = self.W_b(x_b)
            I_a = self.W_a(x_a)

            tau_base = 0.02
            h_b = h_b + self.dt * (-(h_b) / tau_base + I_b)
            h_a = h_a + self.dt * (-(h_a) / tau_base + I_a)

            g_eff = torch.abs(h_b) + 0.5 * torch.abs(h_a)
            tau_dynamic = self.tau_m_min + (self.tau_m_max - self.tau_m_min) * torch.exp(-g_eff)
            tau_dynamic = torch.clamp(tau_dynamic, self.tau_m_min, self.tau_m_max)

            A = A + self.dt * (-A / self.adapt_tau)

            theta = self.theta_base + self.adapt_gain * A

            drive = h_b + 0.7 * h_a
            dV = (-(V) + drive) / tau_dynamic
            V = V + self.dt * dV

            if self.use_hard_spikes:
                with torch.no_grad():
                    spk_hard = torch.zeros_like(V)
                    burst_mask = V >= self.theta_burst
                    spike_mask = (V >= theta) & (~burst_mask)
                    spk_hard[spike_mask] = 1.0
                    spk_hard[burst_mask] = self.burst_k
                spk_soft = torch.sigmoid(5.0 * (V - theta))
                spikes = spk_hard + (spk_soft - spk_soft.detach())
            else:
                spikes = torch.sigmoid(5.0 * (V - theta))

            reset_mask = spikes > 0.5
            V = torch.where(reset_mask, torch.zeros_like(V), V)

            A = A + self.dt * spikes

            new_state = {"V": V, "A": A, "h_b": h_b, "h_a": h_a}
            return spikes, new_state

    def train_omni_torch(
        steps: int = 1000,
        seq_len: int = 50,
        batch_size: int = 32,
        hidden_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        save_prefix: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = OmniNeuronLayer(
            in_features_basal=1,
            in_features_apical=1,
            hidden_size=hidden_size,
            dt=1e-2,
            use_hard_spikes=False,
        ).to(device)

        readout = nn.Linear(hidden_size, 1).to(device)
        optim = torch.optim.Adam(list(model.parameters()) + list(readout.parameters()), lr=lr)

        def make_batch():
            t = torch.linspace(0, 2 * torch.pi, seq_len + 1).unsqueeze(0).repeat(batch_size, 1)
            phase = torch.rand(batch_size, 1) * 2 * torch.pi
            x = torch.sin(t + phase)
            x_b = x[:, :-1].unsqueeze(-1)
            target = x[:, 1:].unsqueeze(-1)
            return x_b.to(device), target.to(device)

        for step in range(steps):
            x_b, target = make_batch()
            B, T, _ = x_b.shape

            state = model.init_state(B, device=device)
            preds = []

            for t_idx in range(T):
                xt = x_b[:, t_idx, :]
                zeros_a = torch.zeros_like(xt)
                spk, state = model(xt, zeros_a, state)
                out = readout(spk)
                preds.append(out.unsqueeze(1))

            preds = torch.cat(preds, dim=1)
            loss = F.mse_loss(preds, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (step + 1) % 100 == 0:
                msg = f"[OMNI Torch] step {step+1}/{steps}, loss={loss.item():.6f}"
                print(msg)
                if run_id is not None:
                    log_run(run_id, msg, progress=(step + 1) / steps)

        print("Finished Omni Torch training.")
        if save_prefix is not None:
            save_omni_torch(model, readout, f"{save_prefix}_omni_torch")
            if run_id is not None:
                log_run(run_id, f"Saved Omni Torch to prefix {save_prefix}_omni_torch", progress=1.0)
        return model, readout


# ----------------------------------------------------------------------
# Visualization helpers + PNG generators
# ----------------------------------------------------------------------
def plot_raster(mesh: NeuronMesh, title: Optional[str] = None, max_neurons: int = 50):
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available")

    S = mesh.get_spike_matrix()
    if S.size == 0:
        print("No spikes to plot.")
        return

    T_steps, N = S.shape
    N_plot = min(N, max_neurons)
    S = S[:, :N_plot]

    t = np.arange(T_steps) * mesh.dt

    fig, ax = plt.subplots(figsize=(8, 4))
    for n in range(N_plot):
        spk_times = t[S[:, n] > 0]
        ax.vlines(spk_times, n + 0.5, n + 1.5)

    ax.set_ylabel("Neuron index")
    ax.set_xlabel("Time (s)")
    if title is None:
        title = f"Raster ({mesh.neuron_type.name})"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_rate(mesh: NeuronMesh, title: Optional[str] = None, window: float = 0.05):
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available")

    S = mesh.get_spike_matrix()
    if S.size == 0:
        print("No spikes to plot.")
        return

    T_steps, N = S.shape
    t = np.arange(T_steps) * mesh.dt
    win_steps = max(1, int(window / mesh.dt))

    pop_spk = np.mean(S, axis=1) / mesh.dt
    kernel = np.ones(win_steps) / win_steps
    pop_smooth = np.convolve(pop_spk, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, pop_smooth)
    ax.set_ylabel("Rate (Hz)")
    ax.set_xlabel("Time (s)")
    if title is None:
        title = f"Population rate ({mesh.neuron_type.name})"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def mesh_raster_png(mesh: NeuronMesh, max_neurons: int = 50) -> bytes:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available")
    import io

    S = mesh.get_spike_matrix()
    if S.size == 0:
        S = np.zeros((1, mesh.N), dtype=np.float32)
    T_steps, N = S.shape
    N_plot = min(N, max_neurons)
    S = S[:, :N_plot]
    t = np.arange(T_steps) * mesh.dt

    fig, ax = plt.subplots(figsize=(6, 3))
    for n in range(N_plot):
        spk_times = t[S[:, n] > 0]
        ax.vlines(spk_times, n + 0.5, n + 1.5)
    ax.set_ylabel("Neuron index")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Raster ({mesh.neuron_type.name})")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def mesh_rate_png(mesh: NeuronMesh, window: float = 0.05) -> bytes:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available")
    import io

    S = mesh.get_spike_matrix()
    if S.size == 0:
        S = np.zeros((1, mesh.N), dtype=np.float32)
    T_steps, N = S.shape
    t = np.arange(T_steps) * mesh.dt
    win_steps = max(1, int(window / mesh.dt))

    pop_spk = np.mean(S, axis=1) / mesh.dt
    kernel = np.ones(win_steps) / win_steps
    pop_smooth = np.convolve(pop_spk, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, pop_smooth)
    ax.set_ylabel("Rate (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Population rate ({mesh.neuron_type.name})")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_multi_mesh_summary(factory: NeuronFactory):
    stats = factory.summary()
    for name, st in stats.items():
        print(
            f"[{name}] type={st.neuron_type}, N={st.N}, "
            f"mean_rate={st.mean_rate_hz:.2f} Hz, peak_rate={st.peak_rate_hz:.2f} Hz, "
            f"frac_active={st.frac_active:.2f}"
        )

    if not _HAS_MPL:
        return

    labels = list(stats.keys())
    means = [stats[name].mean_rate_hz for name in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, means)
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_title("Mesh mean firing rates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Unsupervised mesh-level training (STDP + homeostasis)
# ----------------------------------------------------------------------
def _generic_unsupervised_training(
    neuron_type: NeuronType,
    N: int = 100,
    dt: float = 1e-3,
    episodes: int = 10,
    T_per_episode: float = 0.5,
    base_rate_hz: float = 5.0,
    modulation_hz: float = 1.0,
    stdp: bool = True,
    homeostasis: bool = True,
    seed: int = 1234,
    preset_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> NeuronMesh:
    mesh = NeuronMesh(neuron_type=neuron_type, N=N, dt=dt)
    apply_preset(mesh, preset_name)
    mesh.randomize_connectivity(p_connect=0.2, w_scale=0.05, seed=seed)
    mesh.enable_stdp(stdp)
    mesh.enable_homeostasis(homeostasis)

    rng = np.random.default_rng(seed)

    steps_per_ep = int(T_per_episode / dt)

    for ep in range(episodes):
        phase = rng.uniform(0, 2 * np.pi)

        for _ in range(steps_per_ep):
            t = mesh.time
            rate = base_rate_hz + base_rate_hz * np.sin(2 * np.pi * modulation_hz * t + phase)
            lam = max(rate, 0.0) * dt
            I_basal = rng.poisson(lam=lam, size=N).astype(np.float32)
            I_apical = np.zeros(N, dtype=np.float32)
            mesh.step(I_basal, I_apical)

        st = mesh.compute_stats()
        msg = (
            f"[TRAIN {neuron_type.name}] ep {ep+1}/{episodes} "
            f"mean_rate={st.mean_rate_hz:.2f} Hz, frac_active={st.frac_active:.2f}"
        )
        print(msg)
        if run_id is not None:
            log_run(run_id, msg, progress=(ep + 1) / episodes, stats=st)

    return mesh


def train_lif_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.LIF, run_id=run_id, preset_name=preset_name, **kwargs)


def train_izh_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.IZH, run_id=run_id, preset_name=preset_name, **kwargs)


def train_glif_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.GLIF, run_id=run_id, preset_name=preset_name, **kwargs)


def train_pyr_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.PYR_L5, run_id=run_id, preset_name=preset_name, **kwargs)


def train_fs_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.INT_FS, run_id=run_id, preset_name=preset_name, **kwargs)


def train_ch_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.INT_CH, run_id=run_id, preset_name=preset_name, **kwargs)


def train_mr_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.INT_MR, run_id=run_id, preset_name=preset_name, **kwargs)


def train_cbl_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.CBL_PK, run_id=run_id, preset_name=preset_name, **kwargs)


def train_thal_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.THAL_G, run_id=run_id, preset_name=preset_name, **kwargs)


def train_omni_mesh(run_id: Optional[str] = None, preset_name: Optional[str] = None, **kwargs) -> NeuronMesh:
    return _generic_unsupervised_training(NeuronType.OMNI, run_id=run_id, preset_name=preset_name, **kwargs)


def train_all_mesh_types(preset_name: Optional[str] = None):
    results: Dict[str, MeshStats] = {}
    trainers = [
        train_lif_mesh,
        train_izh_mesh,
        train_glif_mesh,
        train_pyr_mesh,
        train_fs_mesh,
        train_ch_mesh,
        train_mr_mesh,
        train_cbl_mesh,
        train_thal_mesh,
        train_omni_mesh,
    ]
    for trainer in trainers:
        mesh = trainer(
            N=64,
            episodes=3,
            T_per_episode=0.3,
            base_rate_hz=5.0,
            modulation_hz=1.0,
            preset_name=preset_name,
        )
        st = mesh.compute_stats()
        results[mesh.neuron_type.name] = st
        print(
            f"FINAL [{mesh.neuron_type.name}] mean_rate={st.mean_rate_hz:.2f} Hz, "
            f"peak_rate={st.peak_rate_hz:.2f} Hz, frac_active={st.frac_active:.2f}"
        )
    return results


# ----------------------------------------------------------------------
# SUPERVISED TASKS: CLASSIFICATION + SEQUENCE COPY (MESH-LEVEL)
# ----------------------------------------------------------------------
def _make_pattern_dataset(
    n_patterns: int,
    n_features: int,
    n_classes: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_patterns, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n_patterns,), dtype=np.int32)
    return X, y


def _encode_pattern_to_current(
    pattern: np.ndarray,
    N: int,
    dt: float,
    T: float,
    rate_on: float = 20.0,
    rate_off: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = pattern.shape[0]
    steps = int(T / dt)
    I = np.zeros((steps, N), dtype=np.float32)

    neurons_per_feature = max(1, N // n_features)
    for f in range(n_features):
        idx_start = f * neurons_per_feature
        idx_end = min(N, (f + 1) * neurons_per_feature)
        rate = rate_on if pattern[f] > 0.5 else rate_off
        lam = rate * dt
        I[:, idx_start:idx_end] = rng.poisson(lam=lam, size=(steps, idx_end - idx_start)).astype(np.float32)

    return I


def train_mesh_classifier_for_type(
    neuron_type: NeuronType,
    N: int = 100,
    n_features: int = 8,
    n_classes: int = 3,
    n_patterns: int = 100,
    T: float = 0.3,
    dt: float = 1e-3,
    lr: float = 1e-2,
    epochs: int = 5,
    seed: int = 0,
    save_prefix: Optional[str] = None,
    preset_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[NeuronMesh, np.ndarray]:
    rng = np.random.default_rng(seed)

    mesh = NeuronMesh(neuron_type=neuron_type, N=N, dt=dt)
    apply_preset(mesh, preset_name)
    mesh.randomize_connectivity(p_connect=0.2, w_scale=0.05, seed=seed)

    X, y = _make_pattern_dataset(n_patterns, n_features, n_classes, seed=seed)
    W_out = rng.normal(0.0, 0.1, size=(N, n_classes)).astype(np.float32)

    steps = int(T / dt)

    def one_hot(c: int, C: int) -> np.ndarray:
        v = np.zeros(C, dtype=np.float32)
        v[c] = 1.0
        return v

    for ep in range(epochs):
        idxs = rng.permutation(n_patterns)
        correct = 0

        for idx in idxs:
            pattern = X[idx]
            label = int(y[idx])
            target = one_hot(label, n_classes)

            mesh.reset_state()
            mesh.spike_history.clear()

            I_seq = _encode_pattern_to_current(
                pattern, N=N, dt=dt, T=T, rate_on=20.0, rate_off=2.0, seed=seed + idx
            )

            for t_step in range(steps):
                I_t = I_seq[t_step]
                I_apical = np.zeros(N, dtype=np.float32)
                mesh.step(I_t, I_apical)

            S = mesh.get_spike_matrix()
            r = np.mean(S, axis=0) / dt
            logits = r @ W_out
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs + 1e-8)
            pred = int(np.argmax(probs))
            if pred == label:
                correct += 1

            grad_logits = probs - target
            dW = np.outer(r, grad_logits)
            W_out -= lr * dW.astype(np.float32)

        acc = correct / n_patterns
        msg = f"[CLASSIFIER {neuron_type.name}] epoch {ep+1}/{epochs}, acc={acc:.3f}"
        print(msg)
        if run_id is not None:
            st = mesh.compute_stats()
            log_run(run_id, msg, progress=(ep + 1) / epochs, stats=st)

    if save_prefix is not None:
        save_classifier(mesh, W_out, f"{save_prefix}_classifier_{neuron_type.name}")
        if run_id is not None:
            log_run(run_id, f"Saved classifier to {save_prefix}_classifier_{neuron_type.name}", progress=1.0)

    return mesh, W_out


def train_classifiers_all_types(save_prefix: Optional[str] = None, preset_name: Optional[str] = None):
    types = [
        NeuronType.LIF,
        NeuronType.IZH,
        NeuronType.GLIF,
        NeuronType.PYR_L5,
        NeuronType.INT_FS,
        NeuronType.INT_CH,
        NeuronType.INT_MR,
        NeuronType.CBL_PK,
        NeuronType.THAL_G,
        NeuronType.OMNI,
    ]
    results: Dict[str, float] = {}
    for t in types:
        mesh, W = train_mesh_classifier_for_type(
            neuron_type=t,
            N=80,
            n_features=8,
            n_classes=3,
            n_patterns=80,
            T=0.25,
            dt=1e-3,
            lr=5e-3,
            epochs=3,
            seed=42,
            save_prefix=save_prefix,
            preset_name=preset_name,
        )
        X, y = _make_pattern_dataset(40, 8, 3, seed=999)
        correct = 0
        for i in range(40):
            pattern = X[i]
            label = int(y[i])
            mesh.reset_state()
            steps = int(0.25 / mesh.dt)
            I_seq = _encode_pattern_to_current(pattern, N=mesh.N, dt=mesh.dt, T=0.25, seed=1000 + i)
            for t_step in range(steps):
                mesh.step(I_seq[t_step], np.zeros(mesh.N, dtype=np.float32))
            S = mesh.get_spike_matrix()
            r = np.mean(S, axis=0) / mesh.dt
            logits = r @ W
            pred = int(np.argmax(logits))
            if pred == label:
                correct += 1
        acc = correct / 40.0
        results[t.name] = acc
        print(f"[CLASSIFIER FINAL {t.name}] acc={acc:.3f}")
    return results


# -------- Sequence copy / temporal prediction (mesh-level decoder) -----
def _make_sequence_dataset(
    n_seqs: int,
    seq_len: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_seqs, seq_len)).astype(np.float32)
    return X


def train_mesh_sequence_copy_for_type(
    neuron_type: NeuronType,
    N: int = 100,
    seq_len: int = 30,
    n_seqs: int = 80,
    T_per_step: float = 0.01,
    dt: float = 1e-3,
    lr: float = 1e-2,
    epochs: int = 5,
    delay_steps: int = 1,
    seed: int = 0,
    save_prefix: Optional[str] = None,
    preset_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[NeuronMesh, np.ndarray]:
    rng = np.random.default_rng(seed)

    mesh = NeuronMesh(neuron_type=neuron_type, N=N, dt=dt)
    apply_preset(mesh, preset_name)
    mesh.randomize_connectivity(p_connect=0.2, w_scale=0.05, seed=seed)

    X = _make_sequence_dataset(n_seqs, seq_len, seed=seed)
    steps_per_bit = int(T_per_step / dt)
    W_dec = rng.normal(0.0, 0.1, size=(N, 1)).astype(np.float32)

    def bit_to_rate(bit: float) -> float:
        return 25.0 if bit > 0.5 else 2.0

    for ep in range(epochs):
        idxs = rng.permutation(n_seqs)
        total_loss = 0.0

        for idx in idxs:
            seq = X[idx]
            mesh.reset_state()
            mesh.spike_history.clear()

            spike_rates_per_t = []

            for t_idx in range(seq_len):
                bit = seq[t_idx]
                rate = bit_to_rate(bit)
                lam = rate * dt
                for _ in range(steps_per_bit):
                    I_b = rng.poisson(lam=lam, size=N).astype(np.float32)
                    mesh.step(I_b, np.zeros(N, dtype=np.float32))
                S_window = np.stack(mesh.spike_history[-steps_per_bit:], axis=0)
                r_t = np.mean(S_window, axis=0) / dt
                spike_rates_per_t.append(r_t)

            spike_rates_per_t = np.stack(spike_rates_per_t, axis=0)

            y_true = seq[delay_steps:]
            X_dec = spike_rates_per_t[:-delay_steps]
            T_eff = X_dec.shape[0]

            for t_idx in range(T_eff):
                r_t = X_dec[t_idx]
                y_t = y_true[t_idx]
                y_hat = float(r_t @ W_dec[:, 0])
                y_hat_sig = 1.0 / (1.0 + np.exp(-y_hat))

                grad = (y_hat_sig - y_t) * y_hat_sig * (1.0 - y_hat_sig)
                dW = np.outer(r_t, grad)
                W_dec -= lr * dW.astype(np.float32)
                total_loss += 0.5 * (y_hat_sig - y_t) ** 2

        msg = f"[SEQ {neuron_type.name}] epoch {ep+1}/{epochs}, loss={total_loss / (len(idxs)*seq_len):.4f}"
        print(msg)
        if run_id is not None:
            st = mesh.compute_stats()
            log_run(run_id, msg, progress=(ep + 1) / epochs, stats=st)

    if save_prefix is not None:
        save_seq_decoder(mesh, W_dec, f"{save_prefix}_seq_{neuron_type.name}")
        if run_id is not None:
            log_run(run_id, f"Saved sequence decoder to {save_prefix}_seq_{neuron_type.name}", progress=1.0)

    return mesh, W_dec


def train_seq_all_types(save_prefix: Optional[str] = None, preset_name: Optional[str] = None):
    types = [
        NeuronType.LIF,
        NeuronType.IZH,
        NeuronType.GLIF,
        NeuronType.PYR_L5,
        NeuronType.INT_FS,
        NeuronType.INT_CH,
        NeuronType.INT_MR,
        NeuronType.CBL_PK,
        NeuronType.THAL_G,
        NeuronType.OMNI,
    ]
    results: Dict[str, float] = {}
    for t in types:
        mesh, W_dec = train_mesh_sequence_copy_for_type(
            neuron_type=t,
            N=80,
            seq_len=20,
            n_seqs=60,
            T_per_step=0.01,
            dt=1e-3,
            lr=5e-3,
            epochs=3,
            delay_steps=1,
            seed=123,
            save_prefix=save_prefix,
            preset_name=preset_name,
        )

        X_test = _make_sequence_dataset(20, 20, seed=1000)
        correct = 0
        total = 0
        steps_per_bit = int(0.01 / mesh.dt)
        rng = np.random.default_rng(9999)

        def bit_to_rate(bit: float) -> float:
            return 25.0 if bit > 0.5 else 2.0

        for seq in X_test:
            mesh.reset_state()
            mesh.spike_history.clear()
            spike_rates_per_t = []

            for bit in seq:
                rate = bit_to_rate(bit)
                lam = rate * mesh.dt
                for _ in range(steps_per_bit):
                    I_b = rng.poisson(lam=lam, size=mesh.N).astype(np.float32)
                    mesh.step(I_b, np.zeros(mesh.N, dtype=np.float32))
                S_window = np.stack(mesh.spike_history[-steps_per_bit:], axis=0)
                r_t = np.mean(S_window, axis=0) / mesh.dt
                spike_rates_per_t.append(r_t)

            spike_rates_per_t = np.stack(spike_rates_per_t, axis=0)
            y_true = seq[1:]
            X_dec = spike_rates_per_t[:-1]
            T_eff = X_dec.shape[0]

            for t_idx in range(T_eff):
                r_t = X_dec[t_idx]
                y_t = y_true[t_idx]
                y_hat = float(r_t @ W_dec[:, 0])
                y_hat_sig = 1.0 / (1.0 + np.exp(-y_hat))
                pred_bit = 1.0 if y_hat_sig > 0.5 else 0.0
                if pred_bit == y_t:
                    correct += 1
                total += 1

        acc = correct / max(total, 1)
        results[t.name] = acc
        print(f"[SEQ FINAL {t.name}] bit accuracy={acc:.3f}")
    return results


# ----------------------------------------------------------------------
# RL: Gym-style environment for NeuronMesh
# ----------------------------------------------------------------------
if _HAS_GYM:
    BaseEnv = gym.Env
else:
    class BaseEnv:
        def reset(self):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError


class NeuronMeshEnv(BaseEnv):
    """
    Gym-style environment where an agent controls input current to a NeuronMesh
    to keep firing rate near a target.

    Observation: [mean_rate, last_action]
    Action: scalar in [-1, 1] that modulates base input rate.
    Reward: - (mean_rate - target_rate)^2
    """

    def __init__(
        self,
        neuron_type: NeuronType = NeuronType.OMNI,
        N: int = 50,
        dt: float = 1e-3,
        target_rate_hz: float = 10.0,
        base_rate_hz: float = 5.0,
        episode_len: int = 200,
        seed: int = 0,
        preset_name: Optional[str] = None,
    ):
        self.mesh = NeuronMesh(neuron_type=neuron_type, N=N, dt=dt)
        apply_preset(self.mesh, preset_name)
        self.mesh.randomize_connectivity(p_connect=0.2, w_scale=0.05, seed=seed)
        self.target_rate = target_rate_hz
        self.base_rate = base_rate_hz
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.last_action = 0.0

        if _HAS_GYM:
            self.observation_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1000.0, 1.0]), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.mesh.reset_state()
        self.steps = 0
        self.last_action = 0.0
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        S = self.mesh.get_spike_matrix()
        if S.size == 0:
            mean_rate = 0.0
        else:
            mean_rate = float(np.mean(S[-5:], axis=0).mean() / self.mesh.dt)
        return np.array([mean_rate, self.last_action], dtype=np.float32)

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            a = float(action[0])
        else:
            a = float(action)
        a = np.clip(a, -1.0, 1.0)
        self.last_action = a

        rate = max(self.base_rate + a * self.base_rate, 0.0)
        lam = rate * self.mesh.dt
        I_b = self.rng.poisson(lam=lam, size=self.mesh.N).astype(np.float32)
        self.mesh.step(I_b, np.zeros(self.mesh.N, dtype=np.float32))

        obs = self._get_obs()
        mean_rate = obs[0]
        reward = -((mean_rate - self.target_rate) ** 2) * 1e-4
        self.steps += 1
        done = self.steps >= self.episode_len

        info = {"mean_rate": mean_rate}
        return obs, reward, done, info


def rl_actor_critic_demo(
    neuron_type: NeuronType = NeuronType.OMNI,
    episodes: int = 50,
    gamma: float = 0.99,
    lr: float = 1e-3,
    seed: int = 0,
    save_prefix: Optional[str] = None,
    preset_name: Optional[str] = None,
    run_id: Optional[str] = None,
):
    env = NeuronMeshEnv(
        neuron_type=neuron_type,
        N=40,
        dt=1e-3,
        target_rate_hz=10.0,
        base_rate_hz=5.0,
        episode_len=200,
        seed=seed,
        preset_name=preset_name,
    )

    if not _HAS_TORCH:
        print("PyTorch not available; running random policy RL demo.")
        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                a = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
                obs, r, done, info = env.step(a)
                total_reward += r
            msg = f"[RL RANDOM {neuron_type.name}] ep {ep+1}/{episodes}, R={total_reward:.3f}"
            print(msg)
            if run_id is not None:
                log_run(run_id, msg, progress=(ep + 1) / episodes)
        if run_id is not None:
            RUN_REGISTRY[run_id]["status"] = "done"
            RUN_REGISTRY[run_id]["mesh"] = env.mesh
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int, hidden: int = 64):
            super().__init__()
            self.fc = nn.Linear(obs_dim, hidden)
            self.policy = nn.Linear(hidden, 1)
            self.value = nn.Linear(hidden, 1)

        def forward(self, x):
            h = torch.tanh(self.fc(x))
            mean = torch.tanh(self.policy(h))  # in [-1,1]
            value = self.value(h)
            return mean, value

    model = ActorCritic(obs_dim=2, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        log_probs = []
        values = []
        rewards = []

        done = False
        while not done:
            mean, value = model(obs_t)
            dist = torch.distributions.Normal(mean, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            a_np = action.detach().cpu().numpy().reshape(-1)
            next_obs, r, done, info = env.step(a_np)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(r)

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(-1)

        values_t = torch.cat(values, dim=0)
        log_probs_t = torch.cat(log_probs, dim=0)

        advantage = returns_t - values_t
        value_loss = advantage.pow(2).mean()
        policy_loss = -(log_probs_t * advantage.detach()).mean()

        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        msg = f"[RL AC {neuron_type.name}] ep {ep+1}/{episodes}, R={total_reward:.3f}, loss={loss.item():.4f}"
        print(msg)
        if run_id is not None:
            log_run(run_id, msg, progress=(ep + 1) / episodes)

    if save_prefix is not None and _HAS_TORCH:
        save_actor_critic(model, f"{save_prefix}_rl_{neuron_type.name}")
        if run_id is not None:
            log_run(run_id, f"Saved RL ActorCritic to {save_prefix}_rl_{neuron_type.name}", progress=1.0)
    if run_id is not None:
        RUN_REGISTRY[run_id]["status"] = "done"
        RUN_REGISTRY[run_id]["mesh"] = env.mesh
    return model


# ----------------------------------------------------------------------
# Demo / CLI
# ----------------------------------------------------------------------
def demo_factory(T: float = 1.0, visualize: bool = False, run_id: Optional[str] = None):
    factory = NeuronFactory()

    pyr = factory.add_mesh("pyr", NeuronType.PYR_L5, N=100)
    fs = factory.add_mesh("fs", NeuronType.INT_FS, N=40)
    ch = factory.add_mesh("ch", NeuronType.INT_CH, N=20)
    mr = factory.add_mesh("mr", NeuronType.INT_MR, N=20)
    th = factory.add_mesh("th", NeuronType.THAL_G, N=30)
    omni = factory.add_mesh("omni", NeuronType.OMNI, N=60)

    for mesh in factory.meshes.values():
        mesh.randomize_connectivity(p_connect=0.1, w_scale=0.05)

    factory.connect_meshes("th", "pyr", p_connect=0.3, w_scale=0.05)
    factory.connect_meshes("pyr", "fs", p_connect=0.3, w_scale=0.05)
    factory.connect_meshes("pyr", "ch", p_connect=0.3, w_scale=0.05)
    factory.connect_meshes("pyr", "mr", p_connect=0.3, w_scale=0.05)
    factory.connect_meshes("pyr", "omni", p_connect=0.3, w_scale=0.05)
    factory.connect_meshes("fs", "pyr", p_connect=0.4, w_scale=-0.08)
    factory.connect_meshes("ch", "pyr", p_connect=0.4, w_scale=-0.1)
    factory.connect_meshes("mr", "pyr", p_connect=0.4, w_scale=-0.06)

    rng = np.random.default_rng(1234)

    def th_input(t: float, N: int):
        rate_hz = 5.0 + 5.0 * np.sin(2 * np.pi * 1.0 * t)
        lam = rate_hz * pyr.dt
        I = rng.poisson(lam=lam, size=N).astype(np.float32)
        return I, np.zeros(N, dtype=np.float32)

    input_map = {"th": th_input}

    factory.run_all(T=T, input_map=input_map)

    plot_multi_mesh_summary(factory)
    if visualize:
        plot_raster(pyr, title="PYR L5 Raster")
        plot_rate(omni, title="OMNI Population Rate")

    if run_id is not None:
        # Expose at least one mesh (omni cortex) to API plots
        RUN_REGISTRY[run_id]["mesh"] = omni


def main():
    parser = argparse.ArgumentParser(description="Neuron Factory Full Stack v7")
    parser.add_argument("--demo", action="store_true", help="Run built-in multi-mesh demo")
    parser.add_argument("--T", type=float, default=1.0, help="Simulation duration (demo)")
    parser.add_argument("--viz", action="store_true", help="Show visualizations")

    parser.add_argument("--train-all", action="store_true", help="Unsupervised STDP+homeostasis for all mesh types")
    parser.add_argument("--preset", type=str, default=None, help="Preset name to apply to all meshes (CLI)")

    parser.add_argument("--train-omni-torch", action="store_true", help="Train PyTorch OmniNeuron on temporal task")

    parser.add_argument("--train-classifiers", action="store_true", help="Train mesh-level classifiers for all neuron types")
    parser.add_argument("--train-seq", action="store_true", help="Train mesh-level sequence-copy for all neuron types")

    parser.add_argument("--rl-demo", action="store_true", help="Run RL demo on NeuronMeshEnv")
    parser.add_argument("--rl-type", type=str, default="OMNI", help="Neuron type for RL demo (e.g., OMNI, LIF, PYR_L5)")

    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Base prefix for saving models (classifiers, seq decoders, omni torch, RL). "
             "Example: ./runs/neuron_factory",
    )

    args = parser.parse_args()

    if args.demo:
        demo_factory(T=args.T, visualize=args.viz, run_id=None)
    elif args.train_all:
        train_all_mesh_types(preset_name=args.preset)
    elif args.train_omni_torch:
        if not _HAS_TORCH:
            print("PyTorch not available; cannot train Omni Torch.")
        else:
            train_omni_torch(save_prefix=args.save_prefix)
    elif args.train_classifiers:
        train_classifiers_all_types(save_prefix=args.save_prefix, preset_name=args.preset)
    elif args.train_seq:
        train_seq_all_types(save_prefix=args.save_prefix, preset_name=args.preset)
    elif args.rl_demo:
        try:
            ntype = NeuronType[args.rl_type]
        except KeyError:
            print(f"Unknown neuron type {args.rl_type}, defaulting to OMNI")
            ntype = NeuronType.OMNI
        rl_actor_critic_demo(neuron_type=ntype, episodes=20, save_prefix=args.save_prefix, preset_name=args.preset)
    else:
        print(
            "No action specified.\n"
            "Use one of:\n"
            "  --demo --T 1.0 --viz\n"
            "  --train-all [--preset PRESET_NAME]\n"
            "  --train-omni-torch [--save-prefix path]\n"
            "  --train-classifiers [--save-prefix path --preset PRESET_NAME]\n"
            "  --train-seq [--save-prefix path --preset PRESET_NAME]\n"
            "  --rl-demo --rl-type OMNI [--save-prefix path --preset PRESET_NAME]\n"
            "\n"
            "Or run as an API server with FastAPI/uvicorn:\n"
            "  uvicorn neuron_factory_full_stack_v7:app --reload\n"
        )


# ----------------------------------------------------------------------
# FASTAPI APP + FRONTEND UI
# ----------------------------------------------------------------------
if _HAS_FASTAPI:
    app = FastAPI(title="Neuron Factory API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        # Control panel with presets + plot panes
        html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Neuron Factory Control Panel</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #050712; color: #f5f5f5; margin: 0; padding: 0; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
    h2 { margin-top: 1.5rem; }
    .card { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px 18px; margin: 12px 0; border: 1px solid rgba(255,255,255,0.06); }
    label { display: block; font-size: 0.85rem; opacity: 0.8; margin-bottom: 4px; }
    select, input[type="text"] { width: 100%; padding: 6px 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); background: rgba(3,6,20,0.9); color: #f5f5f5; }
    button { padding: 8px 14px; border-radius: 999px; border: none; cursor: pointer; font-weight: 600; margin-top: 8px;
             background: radial-gradient(circle at 0 0, #3efaff, #7b61ff); color: #050712; }
    button:disabled { opacity: 0.5; cursor: default; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .row > div { flex: 1 1 200px; }
    pre { max-height: 240px; overflow-y: auto; background: rgba(0,0,0,0.4); padding: 8px; border-radius: 6px; font-size: 0.78rem; }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 8px; font-size: 0.8rem; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.12); font-size: 0.75rem; }
    .plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 10px; }
    .plots img { width: 100%; height: auto; border-radius: 8px; background: #000; }
    .plots button { width: 100%; margin-top: 4px; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Neuron Factory – Omni-Neuron Mesh Lab</h1>
  <div style="opacity:0.75;font-size:0.9rem;margin-bottom:10px;">
    Pick a task, neuron type, and preset, hit <b>Start Run</b>, then watch logs, stats, and plots stream below.
  </div>

  <div class="card">
    <h2>Run Configuration</h2>
    <div class="row">
      <div>
        <label for="taskSelect">Task</label>
        <select id="taskSelect"></select>
      </div>
      <div>
        <label for="neuronSelect">Neuron Type (for per-type tasks)</label>
        <select id="neuronSelect"></select>
      </div>
      <div>
        <label for="presetSelect">Preset</label>
        <select id="presetSelect"></select>
      </div>
      <div>
        <label for="prefixInput">Save Prefix (optional)</label>
        <input id="prefixInput" type="text" placeholder="./runs/neuron_factory_demo" />
      </div>
    </div>
    <button id="startBtn">Start Run</button>
    <div style="margin-top:8px;font-size:0.8rem;opacity:0.8;">
      Current Run ID: <span id="runIdLbl">–</span>
    </div>
  </div>

  <div class="card">
    <h2>Run Status</h2>
    <div class="row">
      <div>
        <div class="pill">Status: <span id="statusLbl">–</span></div>
      </div>
      <div>
        <div class="pill">Progress: <span id="progressLbl">0%</span></div>
      </div>
    </div>

    <h3 style="margin-top:1rem;font-size:0.95rem;">Last Stats</h3>
    <div class="stat-grid" id="statsGrid">
      <div>Neuron Type: <b><span id="statNeuron">–</span></b></div>
      <div>N: <b><span id="statN">–</span></b></div>
      <div>Mean Rate (Hz): <b><span id="statMeanRate">–</span></b></div>
      <div>Peak Rate (Hz): <b><span id="statPeakRate">–</span></b></div>
      <div>Frac Active: <b><span id="statFracActive">–</span></b></div>
      <div>Mean V: <b><span id="statMeanV">–</span></b></div>
    </div>

    <h3 style="margin-top:1rem;font-size:0.95rem;">Plots</h3>
    <div class="plots">
      <div>
        <button id="rasterBtn">Refresh Raster</button>
        <img id="rasterImg" alt="Raster Plot" />
      </div>
      <div>
        <button id="rateBtn">Refresh Rate</button>
        <img id="rateImg" alt="Rate Plot" />
      </div>
    </div>

    <h3 style="margin-top:1rem;font-size:0.95rem;">Logs</h3>
    <pre id="logBox"></pre>
  </div>
</div>

<script>
const taskSelect = document.getElementById('taskSelect');
const neuronSelect = document.getElementById('neuronSelect');
const presetSelect = document.getElementById('presetSelect');
const prefixInput = document.getElementById('prefixInput');
const startBtn = document.getElementById('startBtn');
const runIdLbl = document.getElementById('runIdLbl');
const statusLbl = document.getElementById('statusLbl');
const progressLbl = document.getElementById('progressLbl');
const logBox = document.getElementById('logBox');

const statNeuron = document.getElementById('statNeuron');
const statN = document.getElementById('statN');
const statMeanRate = document.getElementById('statMeanRate');
const statPeakRate = document.getElementById('statPeakRate');
const statFracActive = document.getElementById('statFracActive');
const statMeanV = document.getElementById('statMeanV');

const rasterBtn = document.getElementById('rasterBtn');
const rateBtn = document.getElementById('rateBtn');
const rasterImg = document.getElementById('rasterImg');
const rateImg = document.getElementById('rateImg');

let currentRunId = null;
let pollTimer = null;
let presetMap = {};  // neuron_type -> [preset names]

async function loadNeuronTypes() {
  const res = await fetch('/api/neuron_types');
  const data = await res.json();
  neuronSelect.innerHTML = '';
  data.forEach(nt => {
    const opt = document.createElement('option');
    opt.value = nt;
    opt.textContent = nt;
    neuronSelect.appendChild(opt);
  });
}

async function loadTasks() {
  const res = await fetch('/api/tasks');
  const data = await res.json();
  taskSelect.innerHTML = '';
  data.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.id;
    opt.textContent = t.label;
    taskSelect.appendChild(opt);
  });
}

async function loadPresets() {
  const res = await fetch('/api/presets');
  const data = await res.json();
  presetMap = data || {};
  refreshPresetOptions();
}

function refreshPresetOptions() {
  const nt = neuronSelect.value || 'OMNI';
  const presets = presetMap[nt] || [];
  presetSelect.innerHTML = '';
  if (!presets.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(none)';
    presetSelect.appendChild(opt);
    return;
  }
  // Always include "(none)" first
  let optNone = document.createElement('option');
  optNone.value = '';
  optNone.textContent = '(none)';
  presetSelect.appendChild(optNone);
  presets.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    presetSelect.appendChild(opt);
  });
}

async function startRun() {
  const task = taskSelect.value;
  const neuronType = neuronSelect.value;
  const preset = presetSelect.value || null;
  const prefix = prefixInput.value.trim() || null;

  startBtn.disabled = true;
  statusLbl.textContent = 'starting...';
  progressLbl.textContent = '0%';
  logBox.textContent = '';
  rasterImg.src = '';
  rateImg.src = '';

  const res = await fetch('/api/start_run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ task, neuron_type: neuronType, preset_name: preset, save_prefix: prefix })
  });
  const data = await res.json();
  currentRunId = data.run_id;
  runIdLbl.textContent = currentRunId || '–';
  startBtn.disabled = false;

  if (pollTimer) clearInterval(pollTimer);
  if (currentRunId) {
    pollTimer = setInterval(pollStatus, 1000);
  }
}

async function pollStatus() {
  if (!currentRunId) return;
  const res = await fetch('/api/run_status/' + currentRunId);
  if (!res.ok) return;
  const data = await res.json();

  statusLbl.textContent = data.status || '–';
  progressLbl.textContent = ((data.progress || 0) * 100).toFixed(0) + '%';
  logBox.textContent = (data.logs || []).join('\\n');

  const s = data.last_stats;
  if (s) {
    statNeuron.textContent = s.neuron_type;
    statN.textContent = s.N;
    statMeanRate.textContent = s.mean_rate_hz.toFixed(2);
    statPeakRate.textContent = s.peak_rate_hz.toFixed(2);
    statFracActive.textContent = s.frac_active.toFixed(2);
    statMeanV.textContent = s.mean_v.toFixed(2);
  } else {
    statNeuron.textContent = '–';
    statN.textContent = '–';
    statMeanRate.textContent = '–';
    statPeakRate.textContent = '–';
    statFracActive.textContent = '–';
    statMeanV.textContent = '–';
  }

  if (data.status === 'done' || data.status === 'error') {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function refreshRaster() {
  if (!currentRunId) return;
  rasterImg.src = '/api/plot/' + currentRunId + '/raster?ts=' + Date.now();
}

function refreshRate() {
  if (!currentRunId) return;
  rateImg.src = '/api/plot/' + currentRunId + '/rate?ts=' + Date.now();
}

startBtn.addEventListener('click', startRun);
neuronSelect.addEventListener('change', refreshPresetOptions);
rasterBtn.addEventListener('click', refreshRaster);
rateBtn.addEventListener('click', refreshRate);

loadNeuronTypes().then(loadPresets);
loadTasks();
</script>
</body>
</html>
        """
        return HTMLResponse(content=html)

    @app.get("/api/neuron_types")
    async def api_neuron_types():
        return [nt.name for nt in NeuronType]

    @app.get("/api/tasks")
    async def api_tasks():
        return [
            {"id": "demo", "label": "Demo: Multi-Mesh Thalamus→Cortex (OMNI mesh exposed)"},
            {"id": "unsupervised", "label": "Unsupervised: STDP + Homeostasis (single type)"},
            {"id": "classifier", "label": "Supervised: Pattern Classifier (single type)"},
            {"id": "sequence", "label": "Supervised: Sequence Copy (single type)"},
            {"id": "omni_torch", "label": "PyTorch Omni-Neuron Temporal Task"},
            {"id": "rl", "label": "RL: Rate Control Actor-Critic"},
        ]

    @app.get("/api/presets")
    async def api_presets():
        # Return neuron_type -> list of preset names
        return {ntype: list(presets.keys()) for ntype, presets in NEURON_PRESETS.items()}

    def _parse_neuron_type(name: Optional[str]) -> NeuronType:
        if not name:
            return NeuronType.OMNI
        try:
            return NeuronType[name]
        except KeyError:
            return NeuronType.OMNI

    @app.post("/api/start_run")
    async def api_start_run(payload: Dict[str, Any], background: BackgroundTasks):
        task = payload.get("task", "demo")
        neuron_type_name = payload.get("neuron_type")
        preset_name = payload.get("preset_name")
        save_prefix = payload.get("save_prefix")
        ntype = _parse_neuron_type(neuron_type_name)

        run_id = create_run(task, ntype.name, {"save_prefix": save_prefix, "preset_name": preset_name})
        RUN_REGISTRY[run_id]["status"] = "running"
        log_run(run_id, f"Started task={task} neuron_type={ntype.name} preset={preset_name} save_prefix={save_prefix}")

        def run_job():
            try:
                if task == "demo":
                    demo_factory(T=1.0, visualize=False, run_id=run_id)
                    log_run(run_id, "Demo completed.", progress=1.0)
                elif task == "unsupervised":
                    trainer_map = {
                        NeuronType.LIF: train_lif_mesh,
                        NeuronType.IZH: train_izh_mesh,
                        NeuronType.GLIF: train_glif_mesh,
                        NeuronType.PYR_L5: train_pyr_mesh,
                        NeuronType.INT_FS: train_fs_mesh,
                        NeuronType.INT_CH: train_ch_mesh,
                        NeuronType.INT_MR: train_mr_mesh,
                        NeuronType.CBL_PK: train_cbl_mesh,
                        NeuronType.THAL_G: train_thal_mesh,
                        NeuronType.OMNI: train_omni_mesh,
                    }
                    trainer = trainer_map.get(ntype, train_omni_mesh)
                    mesh = trainer(
                        N=64,
                        episodes=5,
                        T_per_episode=0.3,
                        base_rate_hz=5.0,
                        modulation_hz=1.0,
                        preset_name=preset_name,
                        run_id=run_id,
                    )
                    st = mesh.compute_stats()
                    RUN_REGISTRY[run_id]["mesh"] = mesh
                    log_run(run_id, "Unsupervised training finished.", progress=1.0, stats=st)
                elif task == "classifier":
                    mesh, W = train_mesh_classifier_for_type(
                        neuron_type=ntype,
                        N=80,
                        n_features=8,
                        n_classes=3,
                        n_patterns=80,
                        T=0.25,
                        dt=1e-3,
                        lr=5e-3,
                        epochs=5,
                        seed=42,
                        save_prefix=save_prefix,
                        preset_name=preset_name,
                        run_id=run_id,
                    )
                    st = mesh.compute_stats()
                    RUN_REGISTRY[run_id]["mesh"] = mesh
                    log_run(run_id, "Classifier training finished.", progress=1.0, stats=st)
                elif task == "sequence":
                    mesh, W_dec = train_mesh_sequence_copy_for_type(
                        neuron_type=ntype,
                        N=80,
                        seq_len=20,
                        n_seqs=60,
                        T_per_step=0.01,
                        dt=1e-3,
                        lr=5e-3,
                        epochs=5,
                        delay_steps=1,
                        seed=123,
                        save_prefix=save_prefix,
                        preset_name=preset_name,
                        run_id=run_id,
                    )
                    st = mesh.compute_stats()
                    RUN_REGISTRY[run_id]["mesh"] = mesh
                    log_run(run_id, "Sequence training finished.", progress=1.0, stats=st)
                elif task == "omni_torch":
                    if not _HAS_TORCH:
                        log_run(run_id, "PyTorch not available; cannot run Omni Torch.", progress=1.0)
                    else:
                        train_omni_torch(
                            steps=500,
                            seq_len=40,
                            batch_size=16,
                            hidden_size=64,
                            lr=1e-3,
                            device=None,
                            save_prefix=save_prefix,
                            run_id=run_id,
                        )
                elif task == "rl":
                    rl_actor_critic_demo(
                        neuron_type=ntype,
                        episodes=20,
                        gamma=0.99,
                        lr=1e-3,
                        seed=0,
                        save_prefix=save_prefix,
                        preset_name=preset_name,
                        run_id=run_id,
                    )
                else:
                    log_run(run_id, f"Unknown task '{task}'", progress=1.0)
                RUN_REGISTRY[run_id]["status"] = "done"
            except Exception as e:
                RUN_REGISTRY[run_id]["status"] = "error"
                RUN_REGISTRY[run_id]["error"] = str(e)
                log_run(run_id, f"ERROR: {e}", progress=1.0)

        background.add_task(run_job)

        return {"run_id": run_id, "status": "started"}

    @app.get("/api/run_status/{run_id}")
    async def api_run_status(run_id: str):
        if run_id not in RUN_REGISTRY:
            return JSONResponse(status_code=404, content={"error": "run_id not found"})
        entry = RUN_REGISTRY[run_id]
        logs = entry["logs"][-200:]
        return {
            "run_id": run_id,
            "task": entry["task"],
            "neuron_type": entry["neuron_type"],
            "status": entry["status"],
            "progress": entry["progress"],
            "logs": logs,
            "last_stats": entry["last_stats"],
            "error": entry["error"],
        }

    @app.get("/api/plot/{run_id}/raster")
    async def api_plot_raster(run_id: str):
        if run_id not in RUN_REGISTRY:
            return JSONResponse(status_code=404, content={"error": "run_id not found"})
        mesh = RUN_REGISTRY[run_id].get("mesh")
        if mesh is None:
            return JSONResponse(status_code=404, content={"error": "no mesh available for this run"})
        if not _HAS_MPL:
            return JSONResponse(status_code=500, content={"error": "matplotlib not available"})
        try:
            png_bytes = mesh_raster_png(mesh)
            return Response(content=png_bytes, media_type="image/png")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/plot/{run_id}/rate")
    async def api_plot_rate(run_id: str):
        if run_id not in RUN_REGISTRY:
            return JSONResponse(status_code=404, content={"error": "run_id not found"})
        mesh = RUN_REGISTRY[run_id].get("mesh")
        if mesh is None:
            return JSONResponse(status_code=404, content={"error": "no mesh available for this run"})
        if not _HAS_MPL:
            return JSONResponse(status_code=500, content={"error": "matplotlib not available"})
        try:
            png_bytes = mesh_rate_png(mesh)
            return Response(content=png_bytes, media_type="image/png")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    main()
