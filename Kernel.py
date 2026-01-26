#!/usr/bin/env python3
# ======================================================================================
# OCTA · FUSION KERNEL v1.0
# ======================================================================================
# Single-file, production-minded reference kernel that COMBINES:
#   (A) OCTA · CLM (Continuous Learning Model) Kernel:
#       - Memory Arithmetic (MA) irreversible hash-chained op-log (SQLite)
#       - Orthospace (online mean/cov + stabilized eigenbasis projection)
#       - Perfect Attractor Router (inverse-square basin routing)
#       - Benchmark Node gates (accuracy + latency p50/p95/p99 + regression checks)
#       - Canary deployment (current + candidate with deterministic request bucketing)
#       - P3P-ish replication (pragmatic): MA op pull/push + deployment hints
#       - API charging (credits ledger)
#
#   (B) OCTA / P3P CLUSTER v12 + LOCAL AGENT API + CURRICULUM RUNNER:
#       - 3-node localhost mesh (JSONL over TCP)
#       - Per-node MAC chain (Ed25519 if cryptography else HMAC-SHA256)
#       - Model CRDT (vector clock + LWW)
#       - Cortex7 spiking + STDP-like gate
#       - Tiny actor-critic learner
#       - On-device corpus + BM25-ish retrieval + dedupe
#       - Curriculum runner (arXiv + Wikipedia + local folders)
#       - Agent API (stdlib HTTP + SSE)
#
# AND A FUSION HTTP SERVER that exposes BOTH APIs on one port:
#   CLM endpoints (unchanged):
#     GET  /health
#     GET  /metrics
#     GET  /model_current
#     POST /ingest          {"event_id": "...", "x":[...], "meta":{...}}
#     POST /label           {"event_id":"...", "y":0|1, "source":"human"}
#     POST /train_tick      {"max_cycles":1}
#     POST /predict         {"x":[...], "api_key":"user123", "request_id":"optional"}
#     POST /deploy_canary   {"admin_key":"...", "candidate_model_id":"...", "canary_pct":10}
#     POST /credits_issue   {"admin_key":"...", "user":"u1", "credits":1000}
#     POST /credits_balance {"user":"u1"}
#     GET  /sync/status
#     POST /sync/pull
#     POST /sync/push
#
#   Agent endpoints:
#     GET  /api/status
#     GET  /api/events              (SSE stream)
#     POST /api/chat                {"message":"...", "session_id":"optional"}
#     POST /api/ingest              {"kind":"text|url|file|folder", ...}
#     GET  /api/corpus/search?q=...
#     GET  /api/corpus/doc?id=...
#     POST /api/set_rate            {"hz":10.0}
#     POST /api/stimulate           {"target":"nodeA","region":"Region_X","vec":[...]} OR {"text":"..."}
#     GET  /api/ledger_verify
#     POST /api/curriculum/start
#     POST /api/curriculum/stop
#     GET  /api/curriculum/status
#     POST /api/export_actor_c       {"node":"nodeA","actor_index":0,"out_path":"./actor0.h","scale":64}
#
# FUSION BRIDGE (optional but ON by default):
#   - Each ingested doc emits a deterministic CLM event vector (dim_raw) derived from text hash.
#   - This ties the corpus/curriculum stream to CLM’s auditable MA/log + router.
#
# Requirements:
#   - Python 3.10+
#   - numpy
#   - Optional: cryptography (for Ed25519)
#
# Usage:
#   # CLM only
#   python3 octa_fusion_v1.py clm init --db clm.sqlite
#   python3 octa_fusion_v1.py clm simulate --db clm.sqlite --dim 64 --n 20000
#   python3 octa_fusion_v1.py clm serve --db clm.sqlite --dim 64 --port 8080 --admin-key "CHANGE_ME"
#
#   # Cluster + Agent only
#   python3 octa_fusion_v1.py cluster run --agent-port 9090
#
#   # FUSION: single HTTP port serving BOTH CLM + Agent (recommended)
#   python3 octa_fusion_v1.py fusion serve --db clm.sqlite --dim 64 --port 8080 --admin-key "CHANGE_ME" \
#       --agent-enable --agent-port-same --mesh-base 10001 --data-dir ./data_fusion
#
# Notes:
#   - This is a single-process reference kernel. For real deployment split services.
#   - Replication is linear-log pragmatic (no fork handling). Add DAG/forks if you need them.
# ======================================================================================

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import hmac
import http.server
import io
import json
import logging
import math
import os
import random
import re
import socket
import socketserver
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import sqlite3
import numpy as np

# ======================================================================================
# Logging / determinism helpers
# ======================================================================================

LOG = logging.getLogger("octa_fusion")

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s | %(message)s",
    )
    logging.Formatter.converter = time.gmtime

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def json_canon(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj: Any) -> str:
    return sha256_hex(json_canon(obj).encode("utf-8"))

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def stable_u64(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def pct_bucket(request_id: str) -> int:
    return stable_u64(request_id) % 100

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

# ======================================================================================
# SECTION A: CLM (SQLite store + MA log + Orthospace + Attractor + Bench + Canary + API)
# ======================================================================================

CLM_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS events (
  event_id     TEXT PRIMARY KEY,
  ts_utc       TEXT NOT NULL,
  x_json       TEXT NOT NULL,
  meta_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
  label_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id     TEXT NOT NULL,
  ts_utc       TEXT NOT NULL,
  y_int        INTEGER NOT NULL,
  source       TEXT NOT NULL,
  version      INTEGER NOT NULL,
  UNIQUE(event_id, version)
);

CREATE INDEX IF NOT EXISTS idx_labels_event ON labels(event_id);
CREATE INDEX IF NOT EXISTS idx_labels_id ON labels(label_id);

CREATE TABLE IF NOT EXISTS ma_ops (
  op_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc       TEXT NOT NULL,
  prev_hash    TEXT NOT NULL,
  op_json      TEXT NOT NULL,
  op_hash      TEXT NOT NULL,
  UNIQUE(op_hash)
);

CREATE INDEX IF NOT EXISTS idx_ma_ops_hash ON ma_ops(op_hash);
CREATE INDEX IF NOT EXISTS idx_ma_ops_id ON ma_ops(op_id);

CREATE TABLE IF NOT EXISTS trainer_state (
  k            TEXT PRIMARY KEY,
  v            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
  model_id     TEXT PRIMARY KEY,
  ts_utc       TEXT NOT NULL,
  algo         TEXT NOT NULL,
  dim          INTEGER NOT NULL,
  blob_b64     TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  config_json  TEXT NOT NULL,
  data_fingerprint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deployments (
  slot         TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL,
  ts_utc       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credits (
  user_id      TEXT PRIMARY KEY,
  balance      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS credit_ledger (
  entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc       TEXT NOT NULL,
  user_id      TEXT NOT NULL,
  delta        INTEGER NOT NULL,
  reason       TEXT NOT NULL,
  ref_json     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_credit_ledger_user ON credit_ledger(user_id);
"""

class CLMStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def init(self) -> None:
        with self._lock:
            self.conn.executescript(CLM_SCHEMA)
            self.conn.commit()
            if self.get_state("ma_prev_hash", "") == "":
                self.set_state("ma_prev_hash", "0" * 64)
            if self.get_state("canary_pct", "") == "":
                self.set_state("canary_pct", "0")
            if self.get_state("cursor_label_id", "") == "":
                self.set_state("cursor_label_id", "0")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---- trainer state ----
    def set_state(self, k: str, v: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO trainer_state(k,v) VALUES (?,?)", (k, v))
            self.conn.commit()

    def get_state(self, k: str, default: str) -> str:
        with self._lock:
            r = self.conn.execute("SELECT v FROM trainer_state WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default

    # ---- events / labels ----
    def put_event(self, event_id: str, x: List[float], meta: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO events(event_id, ts_utc, x_json, meta_json) VALUES (?,?,?,?)",
                (event_id, utc_now_iso(), json_canon(x), json_canon(meta)),
            )
            self.conn.commit()

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self.conn.execute(
                "SELECT event_id, ts_utc, x_json, meta_json FROM events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        if not r:
            return None
        return {
            "event_id": r["event_id"],
            "ts_utc": r["ts_utc"],
            "x": json.loads(r["x_json"]),
            "meta": json.loads(r["meta_json"]),
        }

    def latest_label_version(self, event_id: str) -> int:
        with self._lock:
            r = self.conn.execute(
                "SELECT MAX(version) AS mv FROM labels WHERE event_id=?",
                (event_id,),
            ).fetchone()
        if not r or r["mv"] is None:
            return 0
        return int(r["mv"])

    def put_label(self, event_id: str, y: int, source: str) -> int:
        with self._lock:
            ver = self.latest_label_version(event_id) + 1
            self.conn.execute(
                "INSERT INTO labels(event_id, ts_utc, y_int, source, version) VALUES (?,?,?,?,?)",
                (event_id, utc_now_iso(), int(y), source, ver),
            )
            self.conn.commit()
        return ver

    def fetch_labeled_since(self, last_label_id: int, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT l.label_id, l.event_id, l.ts_utc, l.y_int, l.source, l.version, e.x_json
                FROM labels l
                JOIN events e ON e.event_id = l.event_id
                WHERE l.label_id > ?
                ORDER BY l.label_id ASC
                LIMIT ?
                """,
                (int(last_label_id), int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "label_id": int(r["label_id"]),
                "event_id": r["event_id"],
                "ts_utc": r["ts_utc"],
                "y": int(r["y_int"]),
                "source": r["source"],
                "version": int(r["version"]),
                "x": json.loads(r["x_json"]),
            })
        return out

    # ---- MA op log ----
    def ma_tail_hash(self) -> str:
        return self.get_state("ma_prev_hash", "0" * 64)

    def ma_op_exists(self, op_hash: str) -> bool:
        with self._lock:
            r = self.conn.execute("SELECT 1 FROM ma_ops WHERE op_hash=? LIMIT 1", (op_hash,)).fetchone()
        return r is not None

    def ma_append_op(self, op: Dict[str, Any]) -> str:
        with self._lock:
            prev_hash = self.ma_tail_hash()
            payload = {"prev": prev_hash, "op": op}
            op_hash = sha256_json(payload)

            self.conn.execute(
                "INSERT OR IGNORE INTO ma_ops(ts_utc, prev_hash, op_json, op_hash) VALUES (?,?,?,?)",
                (utc_now_iso(), prev_hash, json_canon(op), op_hash),
            )

            # advance tail only if op_hash is last row
            r = self.conn.execute("SELECT op_hash FROM ma_ops ORDER BY op_id DESC LIMIT 1").fetchone()
            if r and r["op_hash"] == op_hash:
                self.set_state("ma_prev_hash", op_hash)
            self.conn.commit()
        return op_hash

    def ma_import_op(self, prev_hash: str, op: Dict[str, Any], op_hash: str) -> bool:
        payload = {"prev": prev_hash, "op": op}
        if sha256_json(payload) != op_hash:
            return False
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO ma_ops(ts_utc, prev_hash, op_json, op_hash) VALUES (?,?,?,?)",
                (utc_now_iso(), prev_hash, json_canon(op), op_hash),
            )
            self.conn.commit()
        return True

    def ma_find_op_id(self, op_hash: str) -> Optional[int]:
        with self._lock:
            r = self.conn.execute("SELECT op_id FROM ma_ops WHERE op_hash=? LIMIT 1", (op_hash,)).fetchone()
        return int(r["op_id"]) if r else None

    def ma_pull_since(self, from_hash: str, limit: int) -> List[Dict[str, Any]]:
        from_id = self.ma_find_op_id(from_hash)
        if from_id is None:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT prev_hash, op_json, op_hash FROM ma_ops WHERE op_id > ? ORDER BY op_id ASC LIMIT ?",
                (int(from_id), int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "prev_hash": r["prev_hash"],
                "op": json.loads(r["op_json"]),
                "op_hash": r["op_hash"],
            })
        return out

    # ---- models / deployments ----
    def put_model(self, model_id: str, algo: str, dim: int, blob: bytes,
                  metrics: Dict[str, Any], config: Dict[str, Any], data_fingerprint: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO models(model_id, ts_utc, algo, dim, blob_b64, metrics_json, config_json, data_fingerprint)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (model_id, utc_now_iso(), algo, int(dim),
                 base64.b64encode(blob).decode("ascii"),
                 json_canon(metrics), json_canon(config), data_fingerprint),
            )
            self.conn.commit()

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self.conn.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
        return dict(r) if r else None

    def set_deployment(self, slot: str, model_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO deployments(slot, model_id, ts_utc) VALUES (?,?,?)",
                (slot, model_id, utc_now_iso()),
            )
            self.conn.commit()

    def get_deployment(self, slot: str) -> Optional[str]:
        with self._lock:
            r = self.conn.execute("SELECT model_id FROM deployments WHERE slot=?", (slot,)).fetchone()
        return r["model_id"] if r else None

    # ---- credits ----
    def credits_issue(self, user_id: str, credits: int, reason: str, ref: Dict[str, Any]) -> None:
        with self._lock:
            r = self.conn.execute("SELECT balance FROM credits WHERE user_id=?", (user_id,)).fetchone()
            bal = int(r["balance"]) if r else 0
            bal2 = bal + int(credits)
            self.conn.execute("INSERT OR REPLACE INTO credits(user_id, balance) VALUES (?,?)", (user_id, bal2))
            self.conn.execute(
                "INSERT INTO credit_ledger(ts_utc, user_id, delta, reason, ref_json) VALUES (?,?,?,?,?)",
                (utc_now_iso(), user_id, int(credits), reason, json_canon(ref)),
            )
            self.conn.commit()

    def credits_charge(self, user_id: str, cost: int, reason: str, ref: Dict[str, Any]) -> bool:
        with self._lock:
            r = self.conn.execute("SELECT balance FROM credits WHERE user_id=?", (user_id,)).fetchone()
            bal = int(r["balance"]) if r else 0
            if bal < int(cost):
                return False
            bal2 = bal - int(cost)
            self.conn.execute("INSERT OR REPLACE INTO credits(user_id, balance) VALUES (?,?)", (user_id, bal2))
            self.conn.execute(
                "INSERT INTO credit_ledger(ts_utc, user_id, delta, reason, ref_json) VALUES (?,?,?,?,?)",
                (utc_now_iso(), user_id, -int(cost), reason, json_canon(ref)),
            )
            self.conn.commit()
        return True

    def credits_balance(self, user_id: str) -> int:
        with self._lock:
            r = self.conn.execute("SELECT balance FROM credits WHERE user_id=?", (user_id,)).fetchone()
        return int(r["balance"]) if r else 0


class MemoryArithmetic:
    def __init__(self, store: CLMStore):
        self.store = store

    def trace(self, t: Dict[str, Any]) -> str:
        return self.store.ma_append_op({"type": "ma_trace", "trace": t})

    def log_model_delta(self, delta_blob_b64: str, meta: Dict[str, Any]) -> str:
        op = {"type": "model_delta", "delta_b64": delta_blob_b64, "meta": meta}
        return self.store.ma_append_op(op)


@dataclass
class Orthospace:
    dim: int
    k: int
    alpha: float
    mean: np.ndarray
    cov: np.ndarray
    basis: np.ndarray

    @staticmethod
    def init(dim: int, k: int, alpha: float, seed: int) -> "Orthospace":
        rng = np.random.default_rng(seed)
        mean = np.zeros((dim,), dtype=np.float64)
        cov = np.eye(dim, dtype=np.float64)
        Q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        return Orthospace(dim=dim, k=k, alpha=float(alpha), mean=mean, cov=cov, basis=Q.astype(np.float64))

    def update_stats(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        delta = x - self.mean
        self.mean = self.mean + self.alpha * delta
        self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(delta, delta)

    def recompute_basis(self) -> None:
        vals, vecs = np.linalg.eigh(self.cov)
        idx = np.argsort(vals)[::-1][: self.k]
        B = vecs[:, idx]
        for j in range(B.shape[1]):
            if B[0, j] < 0:
                B[:, j] *= -1
        self.basis = B.astype(np.float64)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        return ((x - self.mean) @ self.basis).astype(np.float64)

    def invariant_check(self) -> Dict[str, Any]:
        sym_err = float(np.max(np.abs(self.cov - self.cov.T)))
        ortho_err = float(np.max(np.abs(self.basis.T @ self.basis - np.eye(self.k))))
        finite = bool(np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.cov)) and np.all(np.isfinite(self.basis)))
        return {"cov_sym_err": sym_err, "basis_ortho_err": ortho_err, "finite": finite}


@dataclass
class Attractor:
    aid: int
    center: np.ndarray
    count: int

class PerfectAttractorRouter:
    def __init__(self, k: int, eps: float = 1e-9):
        self.k = int(k)
        self.eps = float(eps)
        self.attractors: List[Attractor] = []

    def ensure(self, n: int, seed: int) -> None:
        if len(self.attractors) >= n:
            return
        rng = np.random.default_rng(seed)
        while len(self.attractors) < n:
            aid = len(self.attractors)
            c = rng.normal(size=(self.k,)).astype(np.float64)
            self.attractors.append(Attractor(aid=aid, center=c, count=0))

    def route(self, z: np.ndarray) -> Tuple[int, float]:
        z = z.astype(np.float64)
        best_aid = 0
        best_score = -1.0
        for a in self.attractors:
            d2 = float(np.sum((z - a.center) ** 2))
            score = 1.0 / (d2 + self.eps)
            if score > best_score + 1e-18:
                best_score, best_aid = score, a.aid
            elif abs(score - best_score) <= 1e-18 and a.aid < best_aid:
                best_score, best_aid = score, a.aid
        return best_aid, float(best_score)

    def update_center(self, aid: int, z: np.ndarray, lr: float = 0.01) -> None:
        a = self.attractors[aid]
        a.center = (1 - lr) * a.center + lr * z
        a.count += 1


@dataclass
class LogRegHead:
    w: np.ndarray
    b: float

    @staticmethod
    def init(dim: int, rng: np.random.Generator) -> "LogRegHead":
        return LogRegHead(w=rng.normal(0.0, 0.01, size=(dim,)).astype(np.float64), b=0.0)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def step(self, X: np.ndarray, y: np.ndarray, lr: float, l2: float) -> float:
        p = self.proba(X)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        g = (p - y)
        grad_w = (X.T @ g) / X.shape[0] + l2 * self.w
        grad_b = float(np.mean(g))
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        return loss

@dataclass
class BasinEnsemble:
    dim: int
    num_basins: int
    global_head: LogRegHead
    basin_heads: List[LogRegHead]
    blend_global: float

    @staticmethod
    def init(dim: int, num_basins: int, seed: int, blend_global: float) -> "BasinEnsemble":
        rng = np.random.default_rng(seed)
        gh = LogRegHead.init(dim, rng)
        heads = [LogRegHead.init(dim, rng) for _ in range(num_basins)]
        return BasinEnsemble(dim=dim, num_basins=num_basins, global_head=gh, basin_heads=heads, blend_global=float(blend_global))

    def proba(self, X: np.ndarray, basin_id: int) -> np.ndarray:
        pg = self.global_head.proba(X)
        pb = self.basin_heads[basin_id].proba(X)
        a = self.blend_global
        return a * pg + (1 - a) * pb

    def to_bytes(self) -> bytes:
        payload = {
            "dim": self.dim,
            "num_basins": self.num_basins,
            "blend_global": self.blend_global,
            "global": {"w": self.global_head.w.tolist(), "b": self.global_head.b},
            "basins": [{"w": h.w.tolist(), "b": h.b} for h in self.basin_heads],
        }
        return json_canon(payload).encode("utf-8")

    @staticmethod
    def from_bytes(b: bytes) -> "BasinEnsemble":
        p = json.loads(b.decode("utf-8"))
        dim = int(p["dim"])
        nb = int(p["num_basins"])
        a = float(p["blend_global"])
        gh = LogRegHead(w=np.array(p["global"]["w"], dtype=np.float64), b=float(p["global"]["b"]))
        heads = [LogRegHead(w=np.array(h["w"], dtype=np.float64), b=float(h["b"])) for h in p["basins"]]
        if len(heads) != nb:
            raise ValueError("basin head count mismatch")
        return BasinEnsemble(dim=dim, num_basins=nb, global_head=gh, basin_heads=heads, blend_global=a)

class ReplayBuffer:
    def __init__(self, max_items: int, seed: int):
        self.max_items = int(max_items)
        self.rng = random.Random(seed)
        self.items: List[Tuple[np.ndarray, int, int]] = []

    def add(self, z: np.ndarray, y: int, basin_id: int) -> None:
        item = (z.astype(np.float64), int(y), int(basin_id))
        if len(self.items) < self.max_items:
            self.items.append(item)
            return
        j = self.rng.randrange(0, len(self.items) + 1)
        if j < len(self.items):
            self.items[j] = item

    def sample(self, n: int) -> List[Tuple[np.ndarray, int, int]]:
        if not self.items:
            return []
        n = min(int(n), len(self.items))
        return self.rng.sample(self.items, n)

def eval_binary(model: BasinEnsemble, X: np.ndarray, y: np.ndarray, basin_ids: np.ndarray) -> Dict[str, Any]:
    p = np.zeros((X.shape[0],), dtype=np.float64)
    for i in range(X.shape[0]):
        p[i] = float(model.proba(X[i:i+1], int(basin_ids[i]))[0])
    yhat = (p >= 0.5).astype(np.int64)
    acc = float(np.mean(yhat == y))
    eps = 1e-12
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))
    return {"n": int(len(y)), "acc": acc, "logloss": logloss, "precision": precision, "recall": recall, "f1": f1}

def latency_bench(model: BasinEnsemble, X: np.ndarray, basin_ids: np.ndarray, iters: int = 2000) -> Dict[str, Any]:
    n = X.shape[0]
    if n == 0:
        return {"iters": 0}
    rng = np.random.default_rng(17)
    idx = rng.integers(0, n, size=(iters,))
    times = np.zeros((iters,), dtype=np.float64)
    for j in range(iters):
        i = int(idx[j])
        x = X[i:i+1]
        b = int(basin_ids[i])
        t0 = time.perf_counter()
        _ = model.proba(x, b)[0]
        times[j] = (time.perf_counter() - t0) * 1000.0
    return {
        "iters": int(iters),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }

def compress_npz(arrs: Dict[str, np.ndarray]) -> str:
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    raw = buf.getvalue()
    comp = zlib.compress(raw, level=6)
    return base64.b64encode(comp).decode("ascii")

def decompress_npz(b64: str) -> Dict[str, np.ndarray]:
    comp = base64.b64decode(b64.encode("ascii"))
    raw = zlib.decompress(comp)
    buf = io.BytesIO(raw)
    out = {}
    with np.load(buf) as data:
        for k in data.files:
            out[k] = data[k]
    return out

@dataclass
class CLMConfig:
    dim_raw: int
    orth_k: int = 32
    orth_alpha: float = 0.01
    attractors: int = 8
    seed: int = 17

    lr: float = 0.05
    l2: float = 1e-4
    steps_per_cycle: int = 300
    batch_new: int = 64
    replay_max: int = 60000
    replay_batch: int = 96
    fetch_limit: int = 20000

    holdout_frac: float = 0.15
    min_holdout: int = 800
    promote_min_acc: float = 0.70
    promote_min_delta_acc: float = 0.002
    promote_max_p95_ms: float = 2.0
    abort_on_invariant_fail: bool = True

    blend_global: float = 0.5
    log_deltas: bool = True

    cost_per_predict: int = 1
    admin_key: str = "CHANGE_ME_ADMIN_KEY"

    gossip_interval_s: float = 3.0
    sync_batch_limit: int = 500

class CLMKernel:
    def __init__(self, store: CLMStore, cfg: CLMConfig):
        self.store = store
        self.cfg = cfg
        self.ma = MemoryArithmetic(store)

        self.cursor_label_id = int(store.get_state("cursor_label_id", "0"))
        self.canary_pct = int(store.get_state("canary_pct", "0"))

        self.orth = Orthospace.init(dim=cfg.dim_raw, k=cfg.orth_k, alpha=cfg.orth_alpha, seed=cfg.seed)
        self.router = PerfectAttractorRouter(k=cfg.orth_k)
        self.router.ensure(cfg.attractors, seed=cfg.seed)

        self.replay = ReplayBuffer(max_items=cfg.replay_max, seed=cfg.seed)

        self.holdout_Z: Optional[np.ndarray] = None
        self.holdout_y: Optional[np.ndarray] = None
        self.holdout_b: Optional[np.ndarray] = None

        self.current_model_id = store.get_deployment("current")
        self.candidate_model_id = store.get_deployment("candidate")

        self.current_model = self._load_or_init_model(self.current_model_id)
        self.candidate_model = self._load_or_none(self.candidate_model_id)

    def _load_or_init_model(self, model_id: Optional[str]) -> BasinEnsemble:
        if model_id:
            row = self.store.get_model(model_id)
            if row:
                blob = base64.b64decode(row["blob_b64"])
                m = BasinEnsemble.from_bytes(blob)
                LOG.info(f"[clm] loaded model_id={model_id}")
                return m
        LOG.info("[clm] no model deployed; initializing new ensemble")
        return BasinEnsemble.init(dim=self.cfg.orth_k, num_basins=self.cfg.attractors, seed=self.cfg.seed, blend_global=self.cfg.blend_global)

    def _load_or_none(self, model_id: Optional[str]) -> Optional[BasinEnsemble]:
        if not model_id:
            return None
        row = self.store.get_model(model_id)
        if not row:
            return None
        blob = base64.b64decode(row["blob_b64"])
        return BasinEnsemble.from_bytes(blob)

    def _ensure_holdout(self, rows: List[Dict[str, Any]]) -> None:
        if self.holdout_Z is not None:
            return
        Zs: List[np.ndarray] = []
        ys: List[int] = []
        bs: List[int] = []
        for r in rows:
            hid = int(hashlib.sha256(r["event_id"].encode("utf-8")).hexdigest(), 16)
            if (hid % 10_000) < int(self.cfg.holdout_frac * 10_000):
                x = np.array(r["x"], dtype=np.float64)
                z = self.orth.project(x)
                b, _ = self.router.route(z)
                Zs.append(z)
                ys.append(int(r["y"]))
                bs.append(int(b))
        if len(ys) >= self.cfg.min_holdout:
            self.holdout_Z = np.stack(Zs, axis=0)
            self.holdout_y = np.array(ys, dtype=np.int64)
            self.holdout_b = np.array(bs, dtype=np.int64)
            self.ma.trace({"type": "holdout_init", "n": len(ys), "ts": utc_now_iso()})
            LOG.info(f"[clm] holdout initialized n={len(ys)}")

    def _data_fingerprint(self, rows: List[Dict[str, Any]]) -> str:
        ids = [r["label_id"] for r in rows]
        payload = {"label_ids": ids, "ma_tail": self.store.ma_tail_hash(), "salt": "octa_fusion_v1"}
        return sha256_json(payload)

    def _incumbent_metrics(self) -> Dict[str, Any]:
        if not self.current_model_id:
            return {"n": 0, "acc": 0.0, "p95_ms": 0.0}
        row = self.store.get_model(self.current_model_id)
        if not row:
            return {"n": 0, "acc": 0.0, "p95_ms": 0.0}
        return json.loads(row["metrics_json"])

    def _promote_candidate(self, cand: BasinEnsemble, metrics: Dict[str, Any], config: Dict[str, Any], fp: str) -> Optional[str]:
        inc = self._incumbent_metrics()
        inc_acc = float(inc.get("acc", 0.0))
        cand_acc = float(metrics.get("acc", 0.0))
        cand_n = int(metrics.get("n", 0))
        cand_p95 = float(metrics.get("latency", {}).get("p95_ms", 0.0))

        if cand_n < self.cfg.min_holdout:
            return None
        if cand_acc < self.cfg.promote_min_acc:
            return None
        if self.current_model_id and (cand_acc - inc_acc) < self.cfg.promote_min_delta_acc:
            return None
        if cand_p95 > self.cfg.promote_max_p95_ms:
            return None

        blob = cand.to_bytes()
        model_id = sha256_hex(blob)[:24]
        self.store.put_model(
            model_id=model_id,
            algo="octa_fusion_v1_basin_ensemble",
            dim=self.cfg.orth_k,
            blob=blob,
            metrics=metrics,
            config=config,
            data_fingerprint=fp,
        )
        self.store.set_deployment("current", model_id)
        self.current_model_id = model_id
        self.current_model = cand
        self.ma.trace({"type": "promote", "model_id": model_id, "metrics": metrics, "inc_acc": inc_acc})
        return model_id

    def _compute_delta_blob(self, before: BasinEnsemble, after: BasinEnsemble) -> str:
        arrs: Dict[str, np.ndarray] = {}
        arrs["g_w"] = (after.global_head.w - before.global_head.w).astype(np.float64)
        arrs["g_b"] = np.array([after.global_head.b - before.global_head.b], dtype=np.float64)
        for i in range(after.num_basins):
            arrs[f"b{i}_w"] = (after.basin_heads[i].w - before.basin_heads[i].w).astype(np.float64)
            arrs[f"b{i}_b"] = np.array([after.basin_heads[i].b - before.basin_heads[i].b], dtype=np.float64)
        return compress_npz(arrs)

    def train_tick(self, max_cycles: int = 1) -> Dict[str, Any]:
        cycles = 0
        total_new = 0
        last_metrics: Optional[Dict[str, Any]] = None

        while cycles < max_cycles:
            rows = self.store.fetch_labeled_since(self.cursor_label_id, self.cfg.fetch_limit)
            if not rows:
                break

            new_cursor = int(rows[-1]["label_id"])
            total_new += len(rows)

            Z_new: List[np.ndarray] = []
            y_new: List[int] = []
            b_new: List[int] = []

            for r in rows:
                x = np.array(r["x"], dtype=np.float64)
                self.orth.update_stats(x)
                z = self.orth.project(x)
                b, score = self.router.route(z)
                self.router.update_center(b, z, lr=0.01)

                self.ma.trace({
                    "type": "route",
                    "event_id": r["event_id"],
                    "label_id": r["label_id"],
                    "basin": int(b),
                    "score": float(score),
                })

                Z_new.append(z)
                y_new.append(int(r["y"]))
                b_new.append(int(b))
                self.replay.add(z, int(r["y"]), int(b))

            self.orth.recompute_basis()
            inv = self.orth.invariant_check()
            self.ma.trace({"type": "orth_invariants", **inv})

            if self.cfg.abort_on_invariant_fail:
                if (not inv["finite"]) or inv["cov_sym_err"] > 1e-6 or inv["basis_ortho_err"] > 1e-6:
                    self.ma.trace({"type": "abort", "reason": "orth_invariant_fail", "inv": inv})
                    raise RuntimeError(f"Orthospace invariant failure: {inv}")

            self._ensure_holdout(rows)

            Z = np.stack(Z_new, axis=0)
            y = np.array(y_new, dtype=np.float64)
            b = np.array(b_new, dtype=np.int64)

            before = dataclasses.replace(
                self.current_model,
                global_head=LogRegHead(w=self.current_model.global_head.w.copy(), b=float(self.current_model.global_head.b)),
                basin_heads=[LogRegHead(w=h.w.copy(), b=float(h.b)) for h in self.current_model.basin_heads],
            )
            cand = dataclasses.replace(
                before,
                global_head=LogRegHead(w=before.global_head.w.copy(), b=float(before.global_head.b)),
                basin_heads=[LogRegHead(w=h.w.copy(), b=float(h.b)) for h in before.basin_heads],
            )

            rng = np.random.default_rng(self.cfg.seed + cycles)
            losses: List[float] = []
            t0 = time.perf_counter()

            for _ in range(self.cfg.steps_per_cycle):
                bn = min(self.cfg.batch_new, len(y))
                idx = rng.integers(0, len(y), size=(bn,)) if bn > 0 else np.array([], dtype=np.int64)
                xb = [Z[int(i)] for i in idx]
                yb = [float(y[int(i)]) for i in idx]
                bb = [int(b[int(i)]) for i in idx]

                for zr, yr, br in self.replay.sample(self.cfg.replay_batch):
                    xb.append(zr)
                    yb.append(float(yr))
                    bb.append(int(br))

                Xb = np.stack(xb, axis=0)
                Yb = np.array(yb, dtype=np.float64)

                loss_g = cand.global_head.step(Xb, Yb, lr=self.cfg.lr, l2=self.cfg.l2)

                loss_b_sum = 0.0
                bb_arr = np.array(bb, dtype=np.int64)
                for basin_id in range(cand.num_basins):
                    mask = (bb_arr == basin_id)
                    if int(mask.sum()) == 0:
                        continue
                    Xk = Xb[mask]
                    Yk = Yb[mask]
                    loss_b_sum += cand.basin_heads[basin_id].step(Xk, Yk, lr=self.cfg.lr, l2=self.cfg.l2)

                losses.append(float(loss_g + loss_b_sum))

            train_ms = (time.perf_counter() - t0) * 1000.0

            if self.holdout_Z is not None and self.holdout_y is not None and self.holdout_b is not None:
                met = eval_binary(cand, self.holdout_Z, self.holdout_y, self.holdout_b)
                lat = latency_bench(cand, self.holdout_Z, self.holdout_b, iters=min(2000, max(200, self.holdout_Z.shape[0] // 2)))
                met["latency"] = lat
            else:
                met = {"n": 0}

            last_metrics = met
            fp = self._data_fingerprint(rows)

            cfgd = dataclasses.asdict(self.cfg)
            cfgd.update({
                "cycle": cycles,
                "cursor_start": self.cursor_label_id,
                "cursor_end": new_cursor,
                "loss_mean": float(np.mean(losses)) if losses else None,
                "train_ms": float(train_ms),
                "orth_invariants": inv,
                "ma_tail": self.store.ma_tail_hash(),
            })

            promoted = self._promote_candidate(cand, met, cfgd, fp)

            if self.cfg.log_deltas:
                try:
                    delta_b64 = self._compute_delta_blob(before, cand)
                    arrs = decompress_npz(delta_b64)
                    l2n = float(np.sqrt(sum(float(np.sum(a*a)) for a in arrs.values())))
                    self.ma.log_model_delta(delta_b64, {
                        "from_model_id": self.current_model_id,
                        "to_model_id": promoted,
                        "cycle": cycles,
                        "l2_norm": l2n,
                        "train_ms": float(train_ms),
                    })
                except Exception as e:
                    self.ma.trace({"type": "delta_log_error", "err": str(e)})

            self.cursor_label_id = new_cursor
            self.store.set_state("cursor_label_id", str(self.cursor_label_id))

            self.ma.trace({
                "type": "train_tick",
                "new_labels": len(rows),
                "metrics": met,
                "promoted": promoted,
                "cursor_label_id": self.cursor_label_id,
                "train_ms": float(train_ms),
            })

            cycles += 1

        return {
            "cycles": cycles,
            "new_labels": total_new,
            "cursor_label_id": self.cursor_label_id,
            "current_model_id": self.current_model_id,
            "candidate_model_id": self.candidate_model_id,
            "canary_pct": self.canary_pct,
            "last_metrics": last_metrics,
            "ma_tail": self.store.ma_tail_hash(),
        }

    def set_canary(self, candidate_model_id: Optional[str], canary_pct: int) -> None:
        canary_pct = max(0, min(100, int(canary_pct)))
        self.canary_pct = canary_pct
        self.store.set_state("canary_pct", str(canary_pct))

        if candidate_model_id:
            row = self.store.get_model(candidate_model_id)
            if not row:
                raise ValueError("candidate_model_id not found")
            self.store.set_deployment("candidate", candidate_model_id)
            self.candidate_model_id = candidate_model_id
            self.candidate_model = BasinEnsemble.from_bytes(base64.b64decode(row["blob_b64"]))
        else:
            self.candidate_model_id = None
            self.candidate_model = None
            if self.current_model_id:
                self.store.set_deployment("candidate", self.current_model_id)

        self.ma.trace({"type": "canary_set", "candidate_model_id": self.candidate_model_id, "canary_pct": self.canary_pct})

    def choose_model(self, request_id: str) -> Tuple[str, BasinEnsemble]:
        if self.candidate_model_id and self.candidate_model and self.canary_pct > 0:
            if pct_bucket(request_id) < self.canary_pct:
                return self.candidate_model_id, self.candidate_model
        if not self.current_model_id:
            raise RuntimeError("No deployed current model.")
        return self.current_model_id, self.current_model

    def predict(self, x: List[float], request_id: str) -> Dict[str, Any]:
        xv = np.array(x, dtype=np.float64)
        if xv.shape[0] != self.cfg.dim_raw:
            raise ValueError(f"expected dim_raw={self.cfg.dim_raw}, got={xv.shape[0]}")

        model_id, model = self.choose_model(request_id)

        self.orth.update_stats(xv)
        z = self.orth.project(xv)
        basin_id, score = self.router.route(z)

        p = float(model.proba(z.reshape(1, -1), basin_id)[0])
        pred = int(p >= 0.5)

        self.ma.trace({"type": "predict", "request_id": request_id, "model_id": model_id,
                       "basin": int(basin_id), "score": float(score), "proba": p, "pred": pred})

        return {"model_id": model_id, "basin": int(basin_id), "score": float(score), "proba": p, "pred": pred}

def http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 2.0) -> Dict[str, Any]:
    data = json_canon(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get_json(url: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

class CLMGossipLoop(threading.Thread):
    def __init__(self, store: CLMStore, kernel: CLMKernel, peers: List[str], interval_s: float, batch_limit: int):
        super().__init__(daemon=True)
        self.store = store
        self.kernel = kernel
        self.peers = [p.rstrip("/") for p in peers if p.strip()]
        self.interval_s = float(interval_s)
        self.batch_limit = int(batch_limit)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                LOG.debug(f"[clm] gossip tick error: {e}")
            self._stop.wait(self.interval_s)

    def tick(self) -> None:
        if not self.peers:
            return
        local_tail = self.store.ma_tail_hash()

        for peer in self.peers:
            try:
                st = http_get_json(peer + "/sync/status", timeout_s=2.0)
                if not st.get("ok"):
                    continue
                peer_tail = st.get("ma_tail", "")
                if not peer_tail or peer_tail == local_tail:
                    continue

                pulled = http_post_json(peer + "/sync/pull", {"from_hash": local_tail, "limit": self.batch_limit}, timeout_s=3.0)
                ops = pulled.get("ops", [])
                if not ops:
                    pulled2 = http_post_json(peer + "/sync/pull", {"from_hash": "0"*64, "limit": self.batch_limit}, timeout_s=3.0)
                    ops = pulled2.get("ops", [])
                    if not ops:
                        continue

                imported = 0
                for o in ops:
                    ok = self.store.ma_import_op(o["prev_hash"], o["op"], o["op_hash"])
                    if ok:
                        imported += 1

                dep = pulled.get("deployments") or {}
                canary_pct = pulled.get("canary_pct", None)
                if isinstance(dep, dict):
                    cur = dep.get("current")
                    cand = dep.get("candidate")
                    if cur and self.store.get_model(cur):
                        self.store.set_deployment("current", cur)
                        self.kernel.current_model_id = cur
                        self.kernel.current_model = self.kernel._load_or_init_model(cur)
                    if cand and self.store.get_model(cand):
                        self.store.set_deployment("candidate", cand)
                        self.kernel.candidate_model_id = cand
                        self.kernel.candidate_model = self.kernel._load_or_none(cand)
                if canary_pct is not None:
                    try:
                        self.store.set_state("canary_pct", str(int(canary_pct)))
                        self.kernel.canary_pct = int(canary_pct)
                    except Exception:
                        pass

                if imported > 0:
                    self.kernel.ma.trace({"type": "gossip_import", "peer": peer, "imported": imported})

            except (urllib.error.URLError, TimeoutError):
                continue
            except Exception:
                continue

# ======================================================================================
# SECTION B: P3P Cluster v12 + Agent + Curriculum + Retrieval + SSE
# ======================================================================================

# -------- Optional Ed25519 signer --------

class Signer:
    """
    Prefer Ed25519 via `cryptography` if available; otherwise deterministic HMAC-SHA256.
    """
    def __init__(self, secret: bytes, prefer_ed25519: bool = True):
        self.secret = secret
        self.kind = "hmac-sha256"
        self._ed_priv = None
        self._ed_pub = None

        if prefer_ed25519:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
                seed = sha256_hex(secret).encode("utf-8")
                seed32 = hashlib.sha256(seed).digest()
                self._ed_priv = Ed25519PrivateKey.from_private_bytes(seed32)
                self._ed_pub = self._ed_priv.public_key()
                self.kind = "ed25519"
            except Exception:
                self.kind = "hmac-sha256"

    def public_key_b64(self) -> str:
        if self.kind != "ed25519" or self._ed_pub is None:
            return ""
        from cryptography.hazmat.primitives import serialization
        pub_bytes = self._ed_pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(pub_bytes).decode("ascii")

    def sign(self, payload: bytes) -> str:
        if self.kind == "ed25519" and self._ed_priv is not None:
            sig = self._ed_priv.sign(payload)
            return base64.b64encode(sig).decode("ascii")
        mac = hmac.new(self.secret, payload, hashlib.sha256).digest()
        return base64.b64encode(mac).decode("ascii")

    def verify(self, payload: bytes, sig_b64: str) -> bool:
        try:
            sig = base64.b64decode(sig_b64.encode("ascii"))
        except Exception:
            return False

        if self.kind == "ed25519" and self._ed_pub is not None:
            try:
                self._ed_pub.verify(sig, payload)
                return True
            except Exception:
                return False

        mac = hmac.new(self.secret, payload, hashlib.sha256).digest()
        return hmac.compare_digest(mac, sig)

# -------- MAC chain --------

@dataclass
class MACEntry:
    rid: str
    kind: str
    gt: int
    ts_ms: int
    payload: Dict[str, Any]
    prev_hash: str
    entry_hash: str
    sig_kind: str
    sig: str
    pub: str = ""

class MACChain:
    def __init__(self, path: str, signer: Signer, node: str):
        self.path = path
        self.signer = signer
        self.node = node
        self._head = "GENESIS"
        self._count = 0
        self._lock = threading.Lock()

        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        self._head = obj.get("entry_hash", self._head)
                        self._count += 1
            except Exception:
                pass

    @property
    def head(self) -> str:
        return self._head

    @property
    def count(self) -> int:
        return self._count

    def append(self, kind: str, gt: int, payload: Dict[str, Any]) -> MACEntry:
        with self._lock:
            ts = now_ms()
            rid = sha256_hex(json_canon({
                "node": self.node,
                "kind": kind,
                "gt": gt,
                "ts": ts,
                "payload": payload,
                "prev": self._head,
            }).encode("utf-8"))[:32]

            body = {
                "rid": rid,
                "kind": kind,
                "gt": gt,
                "ts_ms": ts,
                "payload": payload,
                "prev_hash": self._head,
                "node": self.node,
            }
            body_bytes = json_canon(body).encode("utf-8")
            entry_hash = sha256_hex(body_bytes)
            sig = self.signer.sign(entry_hash.encode("utf-8"))

            entry = MACEntry(
                rid=rid,
                kind=kind,
                gt=gt,
                ts_ms=ts,
                payload=payload,
                prev_hash=self._head,
                entry_hash=entry_hash,
                sig_kind=self.signer.kind,
                sig=sig,
                pub=self.signer.public_key_b64(),
            )

            record = dataclasses.asdict(entry)
            record["node"] = self.node

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json_canon(record) + "\n")

            self._head = entry_hash
            self._count += 1
            return entry

    def verify_chain(self, max_lines: Optional[int] = None) -> Tuple[bool, str]:
        if not os.path.exists(self.path):
            return True, "no-ledger"

        prev = "GENESIS"
        n = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_lines is not None and n >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    if obj.get("prev_hash") != prev:
                        return False, f"prev-hash-mismatch at line {n+1}"

                    entry_hash = obj.get("entry_hash", "")
                    sig = obj.get("sig", "")
                    if not entry_hash or not sig:
                        return False, f"missing-hash-or-sig at line {n+1}"

                    if not self.signer.verify(entry_hash.encode("utf-8"), sig):
                        return False, f"signature-failed at line {n+1}"

                    prev = entry_hash
                    n += 1
            return True, f"ok:{n}"
        except Exception as e:
            return False, f"verify-exception:{e}"

# -------- Mesh (JSONL framed TCP) --------

@dataclass
class MeshMessage:
    typ: str
    src: str
    dst: str
    mid: str
    gt: int
    payload: Dict[str, Any]

class Mesh:
    def __init__(self, node: str, host: str, port: int, peers: List[Tuple[str, int]]):
        self.node = node
        self.host = host
        self.port = port
        self.peers = peers
        self._server: Optional[asyncio.AbstractServer] = None
        self._handlers: Dict[str, Any] = {}
        self._seen: Dict[str, int] = {}
        self._seen_lock = threading.Lock()

    def on(self, typ: str, handler):
        self._handlers[typ] = handler

    async def start(self):
        self._server = await asyncio.start_server(self._handle_conn, self.host, self.port)
        asyncio.create_task(self._hello_loop())

    async def _hello_loop(self):
        await asyncio.sleep(0.2)
        for (h, p) in self.peers:
            try:
                await self.send(h, p, MeshMessage(
                    typ="hello",
                    src=self.node,
                    dst="*",
                    mid=self._new_mid("hello"),
                    gt=-1,
                    payload={"addr": f"{self.host}:{self.port}"},
                ))
            except Exception:
                pass

    def _new_mid(self, salt: str) -> str:
        return sha256_hex(json_canon({
            "node": self.node,
            "salt": salt,
            "t": now_ms(),
            "r": random.random(),
        }).encode("utf-8"))[:24]

    def _dedup(self, mid: str) -> bool:
        ts = now_ms()
        with self._seen_lock:
            if mid in self._seen:
                return False
            self._seen[mid] = ts
            if len(self._seen) > 5000:
                cutoff = ts - 60_000
                for k in list(self._seen.keys())[:1000]:
                    if self._seen.get(k, ts) < cutoff:
                        self._seen.pop(k, None)
            return True

    async def send(self, host: str, port: int, msg: MeshMessage):
        reader, writer = await asyncio.open_connection(host, port)
        writer.write((json_canon(dataclasses.asdict(msg)) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def gossip(self, msg: MeshMessage):
        for (h, p) in self.peers:
            try:
                await self.send(h, p, msg)
            except Exception:
                pass

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8"))
                    msg = MeshMessage(
                        typ=obj["typ"],
                        src=obj["src"],
                        dst=obj.get("dst", "*"),
                        mid=obj["mid"],
                        gt=int(obj.get("gt", -1)),
                        payload=obj.get("payload", {}),
                    )
                except Exception:
                    continue

                if not self._dedup(msg.mid):
                    continue

                handler = self._handlers.get(msg.typ)
                if handler is not None:
                    await handler(msg)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

# -------- Model CRDT (vector clock + LWW) --------

@dataclass
class VC:
    clock: Dict[str, int]

    def bump(self, node: str):
        self.clock[node] = int(self.clock.get(node, 0)) + 1

    def merge(self, other: "VC"):
        for k, v in other.clock.items():
            if v > int(self.clock.get(k, 0)):
                self.clock[k] = int(v)

    def dominates(self, other: "VC") -> bool:
        for k, v in other.clock.items():
            if int(self.clock.get(k, 0)) < int(v):
                return False
        return True

@dataclass
class LWWRegister:
    value: float
    vc: VC
    ts_ms: int
    node: str

class ModelCRDT:
    def __init__(self, node: str, dim: int):
        self.node = node
        self.dim = dim
        self.vc = VC(clock={node: 0})
        self.regs: List[LWWRegister] = [
            LWWRegister(0.0, VC(clock={node: 0}), 0, node) for _ in range(dim)
        ]
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dim": self.dim,
                "vc": dict(self.vc.clock),
                "regs": [
                    {"value": r.value, "vc": dict(r.vc.clock), "ts_ms": r.ts_ms, "node": r.node}
                    for r in self.regs
                ],
            }

    def set_vec(self, vec: List[float]):
        if len(vec) != self.dim:
            raise ValueError("dim mismatch")
        with self._lock:
            self.vc.bump(self.node)
            ts = now_ms()
            for i, v in enumerate(vec):
                self.regs[i] = LWWRegister(float(v), VC(clock=dict(self.vc.clock)), ts, self.node)

    def get_vec(self) -> List[float]:
        with self._lock:
            return [r.value for r in self.regs]

    def merge(self, other: Dict[str, Any]) -> bool:
        changed = False
        with self._lock:
            o_vc = VC(clock=dict(other.get("vc", {})))
            self.vc.merge(o_vc)

            o_regs = other.get("regs", [])
            if len(o_regs) != self.dim:
                return False

            for i in range(self.dim):
                o = o_regs[i]
                orr = LWWRegister(
                    value=float(o["value"]),
                    vc=VC(clock=dict(o.get("vc", {}))),
                    ts_ms=int(o.get("ts_ms", 0)),
                    node=str(o.get("node", "")),
                )
                rr = self.regs[i]

                take_other = False
                if orr.vc.dominates(rr.vc):
                    take_other = True
                elif rr.vc.dominates(orr.vc):
                    take_other = False
                else:
                    if orr.ts_ms > rr.ts_ms:
                        take_other = True
                    elif orr.ts_ms < rr.ts_ms:
                        take_other = False
                    else:
                        take_other = (orr.node > rr.node)

                if take_other:
                    self.regs[i] = orr
                    changed = True

        return changed

# -------- Cortex7 (spiking core) --------

@dataclass
class Synapse:
    w: float
    last_update_gt: int

class Cortex7:
    def __init__(self, node: str, n_neurons: int = 4):
        self.node = node
        self.n = n_neurons
        self.v = [0.0 for _ in range(self.n)]
        self.theta = [1.0, 1.2, 1.3, 1.5][:self.n] + [1.0 for _ in range(max(0, self.n-4))]
        self.last_spike_gt = [-10**9 for _ in range(self.n)]
        self.spike_traces: List[List[int]] = [[] for _ in range(self.n)]
        self.reward_mod = 0.0
        self.plasticity = 0.10
        self.decay = 0.92
        self.refractory = 1

        self.syn: Dict[Tuple[int, int], Synapse] = {}
        edges = [(0,1,0.6),(1,2,0.5),(2,3,0.4),(1,0,0.2),(2,1,0.15),(3,2,0.1)]
        for (a,b,w) in edges:
            if a < self.n and b < self.n:
                self.syn[(a,b)] = Synapse(w=w, last_update_gt=0)
        for i in range(self.n):
            self.syn[(i,i)] = Synapse(w=0.05, last_update_gt=0)

    def set_modulators(self, reward_mod: float, plasticity: float, threshold_delta: float):
        self.reward_mod = clamp(reward_mod, -1.0, 1.0)
        self.plasticity = clamp(plasticity, 0.0, 0.35)
        for i in range(self.n):
            self.theta[i] = clamp(self.theta[i] + threshold_delta, 0.5, 3.0)

    def stimulate(self, vec: List[float]):
        for i in range(min(self.n, len(vec))):
            self.v[i] += float(vec[i])

    def step(self, gt: int) -> List[int]:
        for i in range(self.n):
            self.v[i] *= self.decay

        for (a,b), syn in list(self.syn.items()):
            if self.last_spike_gt[a] == gt - 1:
                self.v[b] += syn.w

        spikes: List[int] = []
        for i in range(self.n):
            if gt - self.last_spike_gt[i] <= self.refractory:
                continue
            if self.v[i] >= self.theta[i]:
                spikes.append(i)
                self.last_spike_gt[i] = gt
                self.spike_traces[i].append(gt)
                if len(self.spike_traces[i]) > 32:
                    self.spike_traces[i] = self.spike_traces[i][-32:]
                self.v[i] = 0.0

        if spikes:
            gate = self.reward_mod
            lr = self.plasticity * (0.25 + 0.75*abs(gate))
            for post in spikes:
                for pre in range(self.n):
                    if pre == post:
                        continue
                    if self.last_spike_gt[pre] == gt - 1 and (pre, post) in self.syn:
                        dw = lr * (0.01 + 0.04*gate)
                        self.syn[(pre, post)].w = clamp(self.syn[(pre, post)].w + dw, -1.5, 1.5)
                        self.syn[(pre, post)].last_update_gt = gt
                    if self.last_spike_gt[pre] == gt + 1 and (post, pre) in self.syn:
                        dw = lr * (0.01 + 0.04*gate)
                        self.syn[(post, pre)].w = clamp(self.syn[(post, pre)].w - dw, -1.5, 1.5)
                        self.syn[(post, pre)].last_update_gt = gt

        return spikes

    def state(self) -> Dict[str, Any]:
        return {
            "v": list(self.v),
            "threshold": list(self.theta),
            "reward_mod": self.reward_mod,
            "plasticity": self.plasticity,
            "synapses": {f"{a}->{b}": {"w": s.w, "last_update_gt": s.last_update_gt} for (a,b), s in self.syn.items()},
        }

# -------- Tiny Actor-Critic --------

def softmax(logits: List[float]) -> List[float]:
    m = max(logits) if logits else 0.0
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]

def sample_categorical(ps: List[float], rng: random.Random) -> int:
    r = rng.random()
    c = 0.0
    for i, p in enumerate(ps):
        c += p
        if r <= c:
            return i
    return len(ps) - 1

@dataclass
class ACParams:
    Wp: List[List[float]]
    bp: List[float]
    Wv: List[float]
    bv: float

class TinyActorCritic:
    def __init__(self, in_dim: int, n_actions: int, rng: random.Random):
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.p = ACParams(
            Wp=[[rng.uniform(-0.05, 0.05) for _ in range(in_dim)] for _ in range(n_actions)],
            bp=[0.0 for _ in range(n_actions)],
            Wv=[rng.uniform(-0.05, 0.05) for _ in range(in_dim)],
            bv=0.0,
        )

    def forward(self, x: List[float]) -> Tuple[List[float], float]:
        logits = []
        for a in range(self.n_actions):
            s = self.p.bp[a]
            wa = self.p.Wp[a]
            for i in range(self.in_dim):
                s += wa[i] * x[i]
            logits.append(s)
        v = self.p.bv
        for i in range(self.in_dim):
            v += self.p.Wv[i] * x[i]
        ps = softmax(logits)
        return ps, v

    def update(self, x: List[float], a: int, adv: float, target_v: float, lr: float, clip: float = 1.0):
        ps, v = self.forward(x)
        adv = clamp(adv, -clip, clip)
        dv = clamp(target_v - v, -clip, clip)

        for k in range(self.n_actions):
            g = (1.0 if k == a else 0.0) - ps[k]
            for i in range(self.in_dim):
                self.p.Wp[k][i] += lr * adv * g * x[i]
            self.p.bp[k] += lr * adv * g

        for i in range(self.in_dim):
            self.p.Wv[i] += lr * dv * x[i]
        self.p.bv += lr * dv

# -------- Task Environment --------

class PhaseTask:
    def __init__(self, rng: random.Random, reward_mode: str = "abs"):
        self.rng = rng
        self.t = 0
        self.phase = 0.0
        self.reward_mode = reward_mode
        self.A = 0.8
        self.omega = 0.07
        self.phi0 = rng.uniform(0.0, 2*math.pi)
        self.deltas = [-0.12, 0.0, +0.12]

    def target(self) -> float:
        return self.A * math.sin(self.omega * self.t + self.phi0)

    def step(self, action: int) -> Dict[str, float]:
        action = int(action)
        action = 0 if action < 0 else 2 if action > 2 else action
        noise = self.rng.uniform(-0.01, 0.01)
        self.phase += self.deltas[action] + noise
        self.t += 1
        tgt = self.target()
        err = self.phase - tgt
        abs_err = abs(err)
        reward = -(err * err) if self.reward_mode == "sq" else -(abs_err)
        return {"phase": self.phase, "target": tgt, "err": err, "abs_err": abs_err, "reward": reward}

# -------- Metrics --------

class Metrics:
    def __init__(self):
        self.spike_counts: Dict[str, int] = {}
        self.region_spike_counts: Dict[str, int] = {}
        self.last_global_tick: Dict[str, int] = {}
        self.learn_stats: List[Dict[str, Any]] = []
        self.task_stats: List[Dict[str, Any]] = []
        self.clock_ticks: Dict[str, int] = {}
        self._lock = threading.Lock()

    def mark_tick(self, node: str, gt: int):
        with self._lock:
            self.last_global_tick[node] = int(gt)
            self.clock_ticks[node] = int(gt)

    def add_spike(self, node: str, module: str, neuron: int, region: Optional[str]):
        key = f"{node}.{module}.{neuron}"
        with self._lock:
            self.spike_counts[key] = int(self.spike_counts.get(key, 0)) + 1
            if region:
                rkey = f"{node}.{region}"
                self.region_spike_counts[rkey] = int(self.region_spike_counts.get(rkey, 0)) + 1

    def add_learn(self, row: Dict[str, Any]):
        with self._lock:
            self.learn_stats.append(dict(row))

    def add_task(self, row: Dict[str, Any]):
        with self._lock:
            self.task_stats.append(dict(row))

    def dump(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "spike_counts": dict(self.spike_counts),
                "region_spike_counts": dict(self.region_spike_counts),
                "last_global_tick": dict(self.last_global_tick),
                "clock_ticks": dict(self.clock_ticks),
                "learn_stats": list(self.learn_stats),
                "task_stats": list(self.task_stats),
            }

# -------- Node --------

class Node:
    def __init__(
        self,
        name: str,
        host: str,
        mesh_port: int,
        peers: List[Tuple[str, int]],
        dim: int = 6,
        n_actions: int = 3,
        seed: int = 1,
        data_dir: str = "./data_fusion",
    ):
        self.name = name
        self.host = host
        self.mesh_port = mesh_port
        self.peers = peers
        self.dim = dim
        self.n_actions = n_actions

        salt = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
        self.rng = random.Random(seed ^ salt)

        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        self.signer = Signer(secret=(f"{name}-secret-v12".encode("utf-8")), prefer_ed25519=True)
        self.mac = MACChain(path=os.path.join(data_dir, f"{name}_mac.jsonl"), signer=self.signer, node=name)
        self.metrics = Metrics()

        self.model = ModelCRDT(node=name, dim=dim)
        self.cortex = Cortex7(node=name, n_neurons=4)

        self.in_dim = 10
        self.actors: List[TinyActorCritic] = [TinyActorCritic(self.in_dim, n_actions, self.rng) for _ in range(4)]
        self.task = PhaseTask(self.rng, reward_mode="abs")
        self.rate_hz = 10.0

        self._gt = 0
        self._running = False
        self._stim_q: "asyncio.Queue[Tuple[str, List[float]]]" = asyncio.Queue()

        self.mesh = Mesh(node=name, host=host, port=mesh_port, peers=peers)
        self.mesh.on("hello", self._on_hello)
        self.mesh.on("model_snapshot", self._on_model_snapshot)
        self.mesh.on("stim", self._on_stim)
        self.mesh.on("spike", self._on_spike)

        self.loss_ema = 0.6931471805599453
        self.value_loss_ema = 0.0
        self.policy_adv_ema = 0.0
        self.replay = 0
        self._last_region: Optional[str] = None

        self.mac.append("boot", -1, {
            "mesh": f"{host}:{mesh_port}",
            "sig": self.signer.kind,
            "pub": self.signer.public_key_b64(),
            "dim": dim,
            "actions": n_actions,
            "mac_head": self.mac.head,
            "mac_count": self.mac.count,
        })

    @property
    def gt(self) -> int:
        return self._gt

    def status(self) -> Dict[str, Any]:
        return {
            "node": self.name,
            "gt": self._gt,
            "rate_hz": self.rate_hz,
            "mac_head": self.mac.head,
            "mac_count": self.mac.count,
            "sig_kind": self.signer.kind,
            "model_dim": self.dim,
            "cortex": self.cortex.state(),
        }

    def set_rate(self, hz: float):
        self.rate_hz = clamp(float(hz), 0.0, 200.0)
        self.mac.append("set_rate", self._gt, {"rate_hz": self.rate_hz})

    def enqueue_stim(self, region: str, vec: List[float]):
        try:
            self._stim_q.put_nowait((region, [float(x) for x in vec]))
        except Exception:
            pass

    async def start(self):
        await self.mesh.start()
        self._running = True
        await self._broadcast_model()
        asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False

    async def _on_hello(self, msg: MeshMessage):
        return

    async def _on_model_snapshot(self, msg: MeshMessage):
        changed = self.model.merge(msg.payload.get("snap", {}))
        if changed:
            self.mac.append("model_merge", msg.gt, {"from": msg.src, "mid": msg.mid})

    async def _on_stim(self, msg: MeshMessage):
        region = msg.payload.get("region", None)
        vec = msg.payload.get("vec", [])
        self._last_region = str(region) if region else None
        self.cortex.stimulate(vec)
        self.mac.append("stim_in", msg.gt, {"from": msg.src, "region": self._last_region, "vec": vec})

    async def _on_spike(self, msg: MeshMessage):
        return

    async def _broadcast_model(self):
        snap = self.model.snapshot()
        out = MeshMessage(
            typ="model_snapshot",
            src=self.name,
            dst="*",
            mid=sha256_hex(json_canon({"t": now_ms(), "n": self.name, "k": "model"}).encode("utf-8"))[:24],
            gt=self._gt,
            payload={"snap": snap},
        )
        await self.mesh.gossip(out)

    def _build_state_vec(self, task_obs: Dict[str, float]) -> List[float]:
        phase = float(task_obs["phase"])
        tgt = float(task_obs["target"])
        err = float(task_obs["err"])
        abs_err = float(task_obs["abs_err"])

        recent = 0
        for i in range(self.cortex.n):
            recent += sum(1 for t in self.cortex.spike_traces[i] if t >= self._gt - 10)
        spike_rate = recent / (10.0 * max(1, self.cortex.n))

        th0 = self.cortex.theta[0]
        th1 = self.cortex.theta[1] if self.cortex.n > 1 else th0

        mv = self.model.get_vec()
        m0 = mv[0] if len(mv) > 0 else 0.0
        m1 = mv[1] if len(mv) > 1 else 0.0

        return [
            clamp(phase, -3.0, 3.0) / 3.0,
            clamp(tgt, -1.0, 1.0),
            clamp(err, -3.0, 3.0) / 3.0,
            clamp(abs_err, 0.0, 3.0) / 3.0,
            clamp(spike_rate, 0.0, 1.0),
            clamp(th0, 0.5, 3.0) / 3.0,
            clamp(th1, 0.5, 3.0) / 3.0,
            clamp(m0, -1.0, 1.0),
            clamp(m1, -1.0, 1.0),
            1.0,
        ]

    async def _run_loop(self):
        self.mac.append("clock_tick", self._gt, {"node": self.name})
        self.metrics.mark_tick(self.name, self._gt)

        while self._running:
            drained = 0
            while True:
                try:
                    region, vec = self._stim_q.get_nowait()
                except Exception:
                    break
                drained += 1
                self._last_region = region
                self.cortex.stimulate(vec)
                self.mac.append("stim_local", self._gt, {"region": region, "vec": vec})

                stim_msg = MeshMessage(
                    typ="stim",
                    src=self.name,
                    dst="*",
                    mid=sha256_hex(json_canon({"t": now_ms(), "n": self.name, "k": "stim", "d": drained}).encode("utf-8"))[:24],
                    gt=self._gt,
                    payload={"region": region, "vec": vec},
                )
                asyncio.create_task(self.mesh.gossip(stim_msg))

            spikes = self.cortex.step(self._gt)
            for neuron in spikes:
                self.metrics.add_spike(self.name, "cortex7", neuron, self._last_region)
                self.mac.append("cortex_spike", self._gt, {"neuron": neuron, "region": self._last_region})

                spike_msg = MeshMessage(
                    typ="spike",
                    src=self.name,
                    dst="*",
                    mid=sha256_hex(json_canon({"t": now_ms(), "n": self.name, "k": "spike", "u": neuron}).encode("utf-8"))[:24],
                    gt=self._gt,
                    payload={"neuron": neuron, "region": self._last_region},
                )
                asyncio.create_task(self.mesh.gossip(spike_msg))

            task_obs_preview = {
                "phase": self.task.phase,
                "target": self.task.target(),
                "err": self.task.phase - self.task.target(),
                "abs_err": abs(self.task.phase - self.task.target()),
            }
            x = self._build_state_vec(task_obs_preview)

            actions = []
            mods_reward = []
            mods_plast = []
            mods_th = []

            rm = clamp(1.0 - 2.0 * task_obs_preview["abs_err"], -1.0, 1.0)

            for ac in self.actors:
                ps, _v = ac.forward(x)
                a = sample_categorical(ps, self.rng)
                actions.append(a)

                ent = -sum(p * math.log(p + 1e-12) for p in ps)
                plast = clamp(0.05 + 0.25 * (ent / math.log(self.n_actions)) * task_obs_preview["abs_err"], 0.0, 0.35)
                th_delta = clamp((rm * 0.01) - (task_obs_preview["abs_err"] * 0.005), -0.02, 0.02)

                mods_reward.append(rm)
                mods_plast.append(plast)
                mods_th.append(th_delta)

            action = max(set(actions), key=lambda a: actions.count(a))

            self.cortex.set_modulators(
                reward_mod=sum(mods_reward) / len(mods_reward),
                plasticity=sum(mods_plast) / len(mods_plast),
                threshold_delta=sum(mods_th) / len(mods_th),
            )

            task_obs = self.task.step(action)
            self.metrics.add_task({
                "node": self.name,
                "gt": self._gt,
                "reward": task_obs["reward"],
                "phase": task_obs["phase"],
                "target": task_obs["target"],
                "err": task_obs["err"],
                "abs_err": task_obs["abs_err"],
                "action": action,
            })

            gamma = 0.92
            lr = 0.02

            x2 = self._build_state_vec(task_obs)
            _, v2 = self.actors[0].forward(x2)

            advs = []
            v_losses = []
            for k, ac in enumerate(self.actors):
                _ps, v1 = ac.forward(x)
                target_v = task_obs["reward"] + gamma * v2
                adv = target_v - v1
                advs.append(adv)
                v_losses.append((target_v - v1) ** 2)

                ac.update(x, action, adv=adv, target_v=target_v, lr=lr, clip=1.0)

                self.replay += 1
                self.loss_ema = 0.98 * self.loss_ema + 0.02 * (abs(adv) + 0.1 * sum(v_losses) / max(1, len(v_losses)))
                self.value_loss_ema = 0.98 * self.value_loss_ema + 0.02 * (sum(v_losses) / max(1, len(v_losses)))
                self.policy_adv_ema = 0.98 * self.policy_adv_ema + 0.02 * adv

                self.metrics.add_learn({
                    "node": self.name,
                    "gt": self._gt,
                    "micro": k,
                    "loss_ema": self.loss_ema,
                    "value_loss_ema": self.value_loss_ema,
                    "policy_adv_ema": self.policy_adv_ema,
                    "replay": self.replay,
                })

            if self._gt % 10 == 0:
                self.mac.append("learn_checkpoint", self._gt, {
                    "loss_ema": self.loss_ema,
                    "value_loss_ema": self.value_loss_ema,
                    "policy_adv_ema": self.policy_adv_ema,
                    "replay": self.replay,
                    "mean_adv": float(sum(advs) / max(1, len(advs))),
                    "mean_vloss": float(sum(v_losses) / max(1, len(v_losses))),
                })

            mv = self.model.get_vec()
            mv[0] = clamp(0.98 * mv[0] + 0.02 * (-task_obs["err"]), -1.0, 1.0)
            if self.dim > 1:
                mv[1] = clamp(0.98 * mv[1] + 0.02 * rm, -1.0, 1.0)
            self.model.set_vec(mv)

            if self._gt % 25 == 0 and self.peers:
                await self._broadcast_model()

            self.metrics.mark_tick(self.name, self._gt)
            self.mac.append("clock_tick", self._gt, {"node": self.name})

            self._gt += 1

            if self.rate_hz <= 0.0:
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(1.0 / self.rate_hz)

    def export_actor_c_header(self, actor_index: int, out_path: str, scale: float = 64.0):
        k = int(actor_index)
        if k < 0 or k >= len(self.actors):
            raise ValueError("actor_index out of range")

        ac = self.actors[k]
        Wp = ac.p.Wp
        bp = ac.p.bp
        Wv = ac.p.Wv
        bv = ac.p.bv

        def q(x: float) -> int:
            return int(clamp(round(x * scale), -127, 127))

        Wp_q = [[q(Wp[a][i]) for i in range(self.in_dim)] for a in range(self.n_actions)]
        bp_q = [q(b) for b in bp]
        Wv_q = [q(w) for w in Wv]
        bv_q = q(bv)

        lines = []
        lines.append("// Auto-generated by OCTA/P3P v12 (fusion)\n")
        lines.append("#pragma once\n\n")
        lines.append(f"#define OCTA_IN_DIM {self.in_dim}\n")
        lines.append(f"#define OCTA_N_ACTIONS {self.n_actions}\n")
        lines.append(f"static const float OCTA_QSCALE = {scale:.8f}f;\n\n")

        lines.append("static const int8_t OCTA_WP[OCTA_N_ACTIONS][OCTA_IN_DIM] = {\n")
        for a in range(self.n_actions):
            lines.append("  { " + ", ".join(str(v) for v in Wp_q[a]) + " },\n")
        lines.append("};\n\n")

        lines.append("static const int8_t OCTA_BP[OCTA_N_ACTIONS] = { " + ", ".join(str(v) for v in bp_q) + " };\n\n")
        lines.append("static const int8_t OCTA_WV[OCTA_IN_DIM] = { " + ", ".join(str(v) for v in Wv_q) + " };\n\n")
        lines.append(f"static const int8_t OCTA_BV = {bv_q};\n\n")

        lines.append(r"""
/*
Inference sketch (you supply softmax):
  logits[a] = (dot(OCTA_WP[a], x_q) + OCTA_BP[a]) / OCTA_QSCALE
  value     = (dot(OCTA_WV, x_q) + OCTA_BV) / OCTA_QSCALE
Where x_q[i] = clamp(round(x[i] * OCTA_QSCALE), -127, 127)
*/
""")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

# -------- Corpus + retrieval + dedupe --------

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP = set("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with you your i we
""".split())

def tokenize(text: str) -> List[str]:
    toks = [m.group(0).lower() for m in _WORD.finditer(text)]
    return [t for t in toks if t and t not in STOP and len(t) > 1]

@dataclass
class Doc:
    doc_id: str
    title: str
    source: str
    source_key: str
    ts_ms: int
    text: str
    length: int
    tf: Dict[str, int]
    content_hash: str

class DocIndex:
    def __init__(self, path_jsonl: str):
        self.path = path_jsonl
        self.docs: List[Doc] = []
        self.df: Dict[str, int] = {}
        self.avgdl = 1.0
        self._lock = threading.Lock()

        self._seen_content: Dict[str, int] = {}
        self._seen_sourcekey: Dict[str, int] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    doc = Doc(
                        doc_id=obj["doc_id"],
                        title=obj.get("title", ""),
                        source=obj.get("source", ""),
                        source_key=obj.get("source_key", ""),
                        ts_ms=int(obj.get("ts_ms", 0)),
                        text=obj.get("text", ""),
                        length=int(obj.get("length", 0)),
                        tf=dict(obj.get("tf", {})),
                        content_hash=obj.get("content_hash", ""),
                    )
                    self.docs.append(doc)
            self._rebuild_stats()
        except Exception:
            self.docs = []
            self.df = {}
            self.avgdl = 1.0
            self._seen_content = {}
            self._seen_sourcekey = {}

    def _rebuild_stats(self):
        df: Dict[str, int] = {}
        total_len = 0
        self._seen_content = {}
        self._seen_sourcekey = {}
        for d in self.docs:
            total_len += max(1, d.length)
            if d.content_hash:
                self._seen_content[d.content_hash] = 1
            if d.source_key:
                self._seen_sourcekey[d.source_key] = 1
            for term in d.tf.keys():
                df[term] = int(df.get(term, 0)) + 1
        self.df = df
        self.avgdl = max(1.0, total_len / max(1, len(self.docs)))

    def has(self, content_hash: str, source_key: str) -> bool:
        with self._lock:
            if content_hash and content_hash in self._seen_content:
                return True
            if source_key and source_key in self._seen_sourcekey:
                return True
            return False

    def add_doc(self, title: str, source: str, source_key: str, text: str) -> Optional[str]:
        title = (title or "").strip()[:240]
        source = (source or "").strip()[:240]
        source_key = (source_key or "").strip()[:512]

        norm = re.sub(r"\s+", " ", (text or "")).strip()
        if len(norm) < 500:
            return None

        content_hash = sha256_hex(norm.encode("utf-8"))
        if self.has(content_hash, source_key):
            return None

        toks = tokenize(norm)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = int(tf.get(t, 0)) + 1

        doc_id = sha256_hex(json_canon({
            "title": title,
            "source": source,
            "source_key": source_key,
            "content_hash": content_hash,
        }).encode("utf-8"))[:16]

        doc = Doc(
            doc_id=doc_id,
            title=title or "untitled",
            source=source or "unknown",
            source_key=source_key,
            ts_ms=now_ms(),
            text=norm,
            length=max(1, len(toks)),
            tf=tf,
            content_hash=content_hash,
        )

        with self._lock:
            self.docs.append(doc)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json_canon(dataclasses.asdict(doc)) + "\n")

            self._seen_content[content_hash] = 1
            if source_key:
                self._seen_sourcekey[source_key] = 1

            for term in doc.tf.keys():
                self.df[term] = int(self.df.get(term, 0)) + 1
            self.avgdl = (self.avgdl * (len(self.docs)-1) + doc.length) / max(1, len(self.docs))

        return doc_id

    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for d in reversed(self.docs):
                if d.doc_id == doc_id:
                    return {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "source": d.source,
                        "source_key": d.source_key,
                        "ts_ms": d.ts_ms,
                        "length": d.length,
                        "content_hash": d.content_hash,
                        "text": d.text,
                    }
        return None

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q = tokenize(query or "")
        if not q:
            return []
        with self._lock:
            N = len(self.docs)
            if N == 0:
                return []
            k1 = 1.2
            b = 0.75
            scores: List[Tuple[float, Doc]] = []
            window = self.docs[-8000:]
            for d in window:
                score = 0.0
                dl = float(max(1, d.length))
                for term in q:
                    f = float(d.tf.get(term, 0))
                    if f <= 0:
                        continue
                    n = float(self.df.get(term, 0))
                    idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
                    denom = f + k1 * (1.0 - b + b * (dl / self.avgdl))
                    score += idf * ((f * (k1 + 1.0)) / (denom + 1e-12))
                if score > 0:
                    scores.append((score, d))
            scores.sort(key=lambda x: x[0], reverse=True)
            out = []
            for s, d in scores[:k]:
                out.append({
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "source": d.source,
                    "source_key": d.source_key,
                    "score": float(s),
                    "snippet": d.text[:1200],
                    "ts_ms": d.ts_ms,
                })
            return out

def strip_html_to_text(raw: bytes) -> str:
    try:
        s = raw.decode("utf-8", errors="ignore")
    except Exception:
        s = str(raw)
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_to_stim_vec(text: str, n: int = 4) -> List[float]:
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    out = []
    for i in range(n):
        x = int.from_bytes(h[i*2:i*2+2], "big") / 65535.0
        out.append((x * 2.0) - 1.0)
    return [v * 1.35 for v in out]

# -------- URL safety allowlist + block private/loopback --------

def is_private_ip(ip: str) -> bool:
    try:
        parts = [int(x) for x in ip.split(".")]
        if len(parts) != 4:
            return True
        a,b,c,d = parts
        if a == 10:
            return True
        if a == 127:
            return True
        if a == 169 and b == 254:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
        return False
    except Exception:
        return True

def resolve_and_block_private(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            addr = info[4][0]
            if ":" in addr:
                return False
            if is_private_ip(addr):
                return False
        return True
    except Exception:
        return False

def url_allowed(url: str, allowed_prefixes: Tuple[str, ...]) -> bool:
    try:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https"):
            return False
        if not any(url.startswith(p) for p in allowed_prefixes):
            return False
        host = u.hostname or ""
        if not host:
            return False
        if not resolve_and_block_private(host):
            return False
        return True
    except Exception:
        return False

# -------- Event bus + curriculum runner --------

class EventBus:
    def __init__(self, maxsize: int = 10000):
        import queue
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=maxsize)

    def emit(self, ev: Dict[str, Any]):
        try:
            self.q.put_nowait(ev)
        except Exception:
            pass

class Backoff:
    def __init__(self, base_s: float = 5.0, max_s: float = 300.0):
        self.base = base_s
        self.max = max_s
        self.fail = 0

    def ok(self):
        self.fail = 0

    def next_sleep(self) -> float:
        self.fail += 1
        t = min(self.max, self.base * (2 ** (self.fail - 1)))
        j = random.random() * 0.2 * t
        return t + j

class CurriculumRunner(threading.Thread):
    def __init__(self, agent: "ClusterAgent", interval_s: int = 60, max_per_cycle: int = 2):
        super().__init__(daemon=True)
        self.agent = agent
        self.interval_s = max(10, int(interval_s))
        self.max_per_cycle = max(1, int(max_per_cycle))
        self._stop = threading.Event()
        self._enabled = threading.Event()

        self._arxiv_backoff = Backoff()
        self._wiki_backoff = Backoff()

        self.arxiv_feeds = [
            ("arxiv:cs.LG", "https://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"),
            ("arxiv:math",  "https://export.arxiv.org/api/query?search_query=cat:math&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"),
            ("arxiv:physics", "https://export.arxiv.org/api/query?search_query=cat:physics&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"),
        ]
        self.wikipedia_pages = [
            "Calculus",
            "Linear_algebra",
            "Group_theory",
            "Real_analysis",
            "Complex_analysis",
            "Classical_mechanics",
            "Quantum_mechanics",
            "Electromagnetism",
            "General_relativity",
            "Statistical_mechanics",
            "Machine_learning",
            "Information_theory",
            "Cryptography",
        ]

        self.local_folders: List[str] = []
        self._cycle = 0
        self._last_cycle_ms = 0

    def enable(self, on: bool):
        if on:
            self._enabled.set()
        else:
            self._enabled.clear()

    def stop(self):
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled.is_set(),
            "interval_s": self.interval_s,
            "max_per_cycle": self.max_per_cycle,
            "cycle": self._cycle,
            "last_cycle_ms": self._last_cycle_ms,
            "arxiv_feeds": [n for n,_ in self.arxiv_feeds],
            "wikipedia_pages": list(self.wikipedia_pages),
            "local_folders": list(self.local_folders),
        }

    def add_folder(self, path: str):
        p = os.path.abspath(path)
        if os.path.isdir(p) and p not in self.local_folders:
            self.local_folders.append(p)

    def run(self):
        while not self._stop.is_set():
            if not self._enabled.is_set():
                time.sleep(0.25)
                continue

            self._cycle += 1
            self._last_cycle_ms = now_ms()

            self.agent.bus.emit({
                "type": "thought",
                "tag": "[curriculum]",
                "text": f"cycle {self._cycle} start",
                "metrics": {},
                "witness": "machine · scheduler",
                "ts_ms": now_ms(),
            })

            budget = self.max_per_cycle

            for folder in list(self.local_folders):
                if budget <= 0:
                    break
                try:
                    picked = 0
                    for root, _dirs, files in os.walk(folder):
                        for fn in files:
                            if picked >= 2:
                                break
                            ext = os.path.splitext(fn)[1].lower()
                            if ext not in (".txt", ".md", ".html", ".htm"):
                                continue
                            fp = os.path.join(root, fn)
                            self.agent.enqueue_ingest({
                                "kind": "file",
                                "path": fp,
                                "title": os.path.basename(fp),
                                "source": "local_folder",
                                "source_key": f"file:{fp}",
                            })
                            picked += 1
                            budget -= 1
                            if budget <= 0:
                                break
                        if budget <= 0:
                            break
                except Exception:
                    continue

            if budget > 0:
                try:
                    enq = self._enqueue_arxiv(budget)
                    budget -= enq
                    self._arxiv_backoff.ok()
                except Exception:
                    sleep = self._arxiv_backoff.next_sleep()
                    self.agent.bus.emit({
                        "type": "thought",
                        "tag": "[curriculum]",
                        "text": f"arXiv fetch failed -> backoff {sleep:.1f}s",
                        "metrics": {"backoff_s": sleep},
                        "witness": "machine · network",
                        "ts_ms": now_ms(),
                    })

            if budget > 0:
                try:
                    enq = self._enqueue_wikipedia(budget)
                    budget -= enq
                    self._wiki_backoff.ok()
                except Exception:
                    sleep = self._wiki_backoff.next_sleep()
                    self.agent.bus.emit({
                        "type": "thought",
                        "tag": "[curriculum]",
                        "text": f"wikipedia fetch failed -> backoff {sleep:.1f}s",
                        "metrics": {"backoff_s": sleep},
                        "witness": "machine · network",
                        "ts_ms": now_ms(),
                    })

            self.agent.bus.emit({
                "type": "thought",
                "tag": "[curriculum]",
                "text": f"cycle {self._cycle} end",
                "metrics": {"remaining_budget": budget},
                "witness": "machine · scheduler",
                "ts_ms": now_ms(),
            })

            t_end = time.time() + self.interval_s
            while time.time() < t_end and not self._stop.is_set():
                time.sleep(0.25)

    def _enqueue_arxiv(self, budget: int) -> int:
        enq = 0
        for feed_name, url in self.arxiv_feeds:
            if enq >= budget:
                break
            raw = self.agent.fetch_url(url, kind="arxiv")
            if not raw:
                continue

            root = ET.fromstring(raw)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entries = root.findall("a:entry", ns)
            for e in entries:
                if enq >= budget:
                    break
                aid = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
                title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
                summary = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
                authors = []
                for au in e.findall("a:author", ns):
                    nm = (au.findtext("a:name", default="", namespaces=ns) or "").strip()
                    if nm:
                        authors.append(nm)

                if not aid or not title or not summary:
                    continue

                text = f"{title}\n\nAuthors: {', '.join(authors)}\n\n{summary}"
                self.agent.enqueue_ingest({
                    "kind": "text",
                    "text": text,
                    "title": f"{feed_name} · {title}",
                    "source": "arxiv",
                    "source_key": f"arxiv:{aid}",
                })
                enq += 1

        return enq

    def _enqueue_wikipedia(self, budget: int) -> int:
        enq = 0
        pages = list(self.wikipedia_pages)
        random.shuffle(pages)
        for p in pages:
            if enq >= budget:
                break
            title = p.replace(" ", "_")
            url = "https://en.wikipedia.org/w/api.php?action=parse&prop=text&format=json&page=" + urllib.parse.quote(title)
            raw = self.agent.fetch_url(url, kind="wiki")
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8", errors="ignore"))
            html = obj.get("parse", {}).get("text", {}).get("*", "")
            if not html or len(html) < 800:
                continue
            text = strip_html_to_text(html.encode("utf-8"))
            self.agent.enqueue_ingest({
                "kind": "text",
                "text": text,
                "title": f"wiki · {title}",
                "source": "wikipedia",
                "source_key": f"wiki:{title}",
            })
            enq += 1
        return enq

# -------- Cluster Agent --------

class ClusterAgent:
    def __init__(self, nodes: Dict[str, Node], data_dir: str, clm_bridge: Optional["CLMBridge"] = None):
        import queue
        self.nodes = nodes
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.index = DocIndex(os.path.join(data_dir, "corpus.jsonl"))
        self.bus = EventBus()

        self.ingest_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=5000)
        self._running = True
        self._ingest_thread = threading.Thread(target=self._ingest_worker, daemon=True)
        self._ingest_thread.start()

        self.curriculum = CurriculumRunner(self, interval_s=75, max_per_cycle=3)
        self.curriculum.start()

        self._net_last_fetch_ms: Dict[str, int] = {}
        self._net_lock = threading.Lock()

        self.clm_bridge = clm_bridge

        # Superhead (optional)
        self.superhead_url = os.environ.get("SUPERHEAD_URL", "").strip()
        self.superhead_token = os.environ.get("SUPERHEAD_TOKEN", "").strip()
        self.superhead_type = os.environ.get("SUPERHEAD_TYPE", "").strip().lower()
        self.superhead_model = os.environ.get("SUPERHEAD_MODEL", "llama2").strip()
        self.superhead_proc: Optional[subprocess.Popen] = None
        self.superhead_api_style = ""

        if self.superhead_type and not self.superhead_url:
            if self.superhead_type == "ollama":
                local_port = 11434
                local_url = f"http://127.0.0.1:{local_port}"
                running = False
                try:
                    urllib.request.urlopen(local_url, timeout=1)
                    running = True
                except Exception:
                    pass
                if not running:
                    try:
                        self.superhead_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(5)
                    except Exception as e:
                        print(f"[agent] failed to launch ollama: {e}")
                self.superhead_url = local_url + "/api/chat"
                self.superhead_api_style = "ollama"
            elif self.superhead_type == "llama.cpp":
                superhead_bin = os.environ.get("SUPERHEAD_BIN", "").strip()
                if superhead_bin:
                    local_port = 9001
                    local_url = f"http://127.0.0.1:{local_port}/v1/chat/completions"
                    running = False
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{local_port}", timeout=1)
                        running = True
                    except Exception:
                        pass
                    if not running:
                        args = [superhead_bin, "--model", self.superhead_model, "--port", str(local_port), "--host", "127.0.0.1"]
                        try:
                            self.superhead_proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            time.sleep(10)
                        except Exception as e:
                            print(f"[agent] failed to launch llama.cpp: {e}")
                    self.superhead_url = local_url
                    self.superhead_api_style = "openai"
            elif self.superhead_type == "tgi":
                webui_dir = os.environ.get("SUPERHEAD_WEBUI_DIR", "").strip()
                if webui_dir:
                    local_port = 7861
                    local_url = f"http://127.0.0.1:{local_port}/v1/chat/completions"
                    running = False
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{local_port}", timeout=1)
                        running = True
                    except Exception:
                        pass
                    if not running:
                        args = ["python", "server.py", "--api", "--port", str(local_port), "--model", self.superhead_model]
                        try:
                            self.superhead_proc = subprocess.Popen(args, cwd=webui_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            time.sleep(15)
                        except Exception as e:
                            print(f"[agent] failed to launch text-generation-webui: {e}")
                    self.superhead_url = local_url
                    self.superhead_api_style = "openai"

        self.bus.emit({
            "type": "thought",
            "tag": "[agent]",
            "text": "cluster agent online · curriculum runner available",
            "metrics": {"docs": len(self.index.docs)},
            "witness": "machine · bridge",
            "ts_ms": now_ms(),
        })

    def stop(self):
        self._running = False
        if self.superhead_proc:
            self.superhead_proc.terminate()
            try:
                self.superhead_proc.wait(timeout=5)
            except Exception:
                self.superhead_proc.kill()
        try:
            self.curriculum.stop()
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        return {
            "cluster": {k: n.status() for k, n in self.nodes.items()},
            "corpus_docs": len(self.index.docs),
            "curriculum": self.curriculum.status(),
            "superhead_type": self.superhead_type if self.superhead_url else "",
            "clm_bridge": self.clm_bridge.status() if self.clm_bridge else {"enabled": False},
            "ts_ms": now_ms(),
        }

    def ledger_verify(self) -> Dict[str, Any]:
        out = {}
        for k, n in self.nodes.items():
            ok, msg = n.mac.verify_chain()
            out[k] = {"ok": ok, "msg": msg, "head": n.mac.head, "count": n.mac.count}
        return out

    def set_rate_all(self, hz: float):
        for n in self.nodes.values():
            n.set_rate(hz)
        self.bus.emit({
            "type": "thought",
            "tag": "[control]",
            "text": f"set_rate_all -> {hz:.3f} hz",
            "metrics": {"rate_hz": hz},
            "witness": "human · machine",
            "ts_ms": now_ms(),
        })

    def stimulate(self, region: str, vec: List[float], target: str = "nodeA"):
        if target not in self.nodes:
            target = "nodeA"
        self.nodes[target].enqueue_stim(region, vec)
        self.bus.emit({
            "type": "thought",
            "tag": "[stim]",
            "text": f"{target} <- {region} vec={','.join(f'{v:.2f}' for v in vec[:4])}",
            "metrics": {"target": target, "region": region},
            "witness": "human · machine",
            "ts_ms": now_ms(),
        })

    def export_actor_c(self, node: str, actor_index: int, out_path: str, scale: float):
        if node not in self.nodes:
            raise ValueError("unknown node")
        self.nodes[node].export_actor_c_header(actor_index=actor_index, out_path=out_path, scale=scale)
        self.bus.emit({
            "type": "thought",
            "tag": "[export]",
            "text": f"export_actor_c -> node={node} actor={actor_index} out={out_path}",
            "metrics": {"node": node, "actor_index": actor_index, "out_path": out_path, "scale": scale},
            "witness": "human · machine",
            "ts_ms": now_ms(),
        })

    # -------------------- Ingest plumbing --------------------

    def enqueue_ingest(self, job: Dict[str, Any]):
        try:
            self.ingest_q.put_nowait(dict(job))
        except Exception:
            self.bus.emit({
                "type": "thought",
                "tag": "[ingest]",
                "text": "ingest queue full; dropping",
                "metrics": {"kind": str(job.get("kind", ""))},
                "witness": "machine · backpressure",
                "ts_ms": now_ms(),
            })

    def fetch_url(self, url: str, kind: str = "generic") -> Optional[bytes]:
        """
        Network fetch with:
          - allowlist prefixes (env ALLOWED_URL_PREFIXES or conservative defaults)
          - blocks private/loopback via DNS resolution
          - size cap
          - simple per-host rate limiting
        """
        allowed = os.environ.get("ALLOWED_URL_PREFIXES", "").strip()
        if allowed:
            allowed_prefixes = tuple([p.strip() for p in allowed.split(",") if p.strip()])
        else:
            allowed_prefixes = (
                "https://export.arxiv.org/",
                "https://en.wikipedia.org/",
                "https://www.wikipedia.org/",
            )

        if not url_allowed(url, allowed_prefixes):
            self.bus.emit({
                "type": "thought",
                "tag": "[net]",
                "text": f"blocked url (allowlist/private): {url}",
                "metrics": {"kind": kind},
                "witness": "machine · policy",
                "ts_ms": now_ms(),
            })
            return None

        u = urllib.parse.urlparse(url)
        host = (u.hostname or "").lower()

        with self._net_lock:
            last = int(self._net_last_fetch_ms.get(host, 0))
            if now_ms() - last < 900:
                return None
            self._net_last_fetch_ms[host] = now_ms()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "OCTA-Fusion/1.0 (+https://minespace.us) python-stdlib",
                    "Accept": "*/*",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw = resp.read(2_000_000 + 1)
                if len(raw) > 2_000_000:
                    return None
                return raw
        except Exception:
            return None

    def _ingest_worker(self):
        import queue
        while self._running:
            try:
                job = self.ingest_q.get(timeout=0.25)
            except queue.Empty:
                continue
            except Exception:
                continue

            kind = (job.get("kind") or "").strip().lower()
            try:
                if kind == "text":
                    text = str(job.get("text", ""))
                    title = str(job.get("title", ""))[:240]
                    source = str(job.get("source", "text"))
                    source_key = str(job.get("source_key", ""))

                    doc_id = self.index.add_doc(title=title, source=source, source_key=source_key, text=text)
                    if doc_id:
                        self.bus.emit({
                            "type": "doc",
                            "tag": "[corpus]",
                            "text": f"ingested doc {doc_id} · {title}",
                            "metrics": {"doc_id": doc_id, "source": source},
                            "witness": "machine · ingest",
                            "ts_ms": now_ms(),
                        })
                        if self.clm_bridge:
                            self.clm_bridge.ingest_text(doc_id=doc_id, title=title, source=source, source_key=source_key, text=text)

                elif kind == "url":
                    url = str(job.get("url", "")).strip()
                    title = str(job.get("title", url))[:240]
                    source = str(job.get("source", "url"))
                    source_key = str(job.get("source_key", f"url:{url}"))

                    raw = self.fetch_url(url, kind="url")
                    if not raw:
                        continue
                    # attempt parse
                    txt = strip_html_to_text(raw)
                    doc_id = self.index.add_doc(title=title, source=source, source_key=source_key, text=txt)
                    if doc_id:
                        self.bus.emit({
                            "type": "doc",
                            "tag": "[corpus]",
                            "text": f"ingested url doc {doc_id} · {title}",
                            "metrics": {"doc_id": doc_id, "url": url},
                            "witness": "machine · network",
                            "ts_ms": now_ms(),
                        })
                        if self.clm_bridge:
                            self.clm_bridge.ingest_text(doc_id=doc_id, title=title, source=source, source_key=source_key, text=txt)

                elif kind == "file":
                    path = os.path.abspath(str(job.get("path", "")).strip())
                    if not path or not os.path.isfile(path):
                        continue
                    if os.path.getsize(path) > 2_000_000:
                        continue
                    try:
                        with open(path, "rb") as f:
                            raw = f.read(2_000_000 + 1)
                        if len(raw) > 2_000_000:
                            continue
                        ext = os.path.splitext(path)[1].lower()
                        if ext in (".html", ".htm"):
                            txt = strip_html_to_text(raw)
                        else:
                            txt = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    title = str(job.get("title", os.path.basename(path)))[:240]
                    source = str(job.get("source", "file"))
                    source_key = str(job.get("source_key", f"file:{path}"))

                    doc_id = self.index.add_doc(title=title, source=source, source_key=source_key, text=txt)
                    if doc_id:
                        self.bus.emit({
                            "type": "doc",
                            "tag": "[corpus]",
                            "text": f"ingested file doc {doc_id} · {title}",
                            "metrics": {"doc_id": doc_id, "path": path},
                            "witness": "machine · ingest",
                            "ts_ms": now_ms(),
                        })
                        if self.clm_bridge:
                            self.clm_bridge.ingest_text(doc_id=doc_id, title=title, source=source, source_key=source_key, text=txt)

                elif kind == "folder":
                    path = os.path.abspath(str(job.get("path", "")).strip())
                    if not path or not os.path.isdir(path):
                        continue
                    self.curriculum.add_folder(path)
                    self.bus.emit({
                        "type": "thought",
                        "tag": "[curriculum]",
                        "text": f"folder added: {path}",
                        "metrics": {"path": path},
                        "witness": "human · machine",
                        "ts_ms": now_ms(),
                    })

                else:
                    continue

            except Exception as e:
                self.bus.emit({
                    "type": "thought",
                    "tag": "[ingest]",
                    "text": f"ingest error: {e}",
                    "metrics": {"kind": kind},
                    "witness": "machine · exception",
                    "ts_ms": now_ms(),
                })

    # -------------------- Chat --------------------

    def _call_superhead(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Optional LLM “superhead” bridge:
          - ollama /api/chat
          - openai-style /v1/chat/completions (llama.cpp, TGI, etc.)
        """
        if not self.superhead_url:
            return None

        try:
            if self.superhead_api_style == "ollama":
                payload = {"model": self.superhead_model or "llama2", "messages": messages, "stream": False}
                req = urllib.request.Request(
                    self.superhead_url,
                    data=json_canon(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.superhead_token}"} if self.superhead_token else {"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                msg = obj.get("message", {}) or {}
                return (msg.get("content") or "").strip() or None

            # openai-style
            payload = {"model": self.superhead_model or "gpt-3.5-turbo", "messages": messages, "temperature": 0.2}
            headers = {"Content-Type": "application/json"}
            if self.superhead_token:
                headers["Authorization"] = f"Bearer {self.superhead_token}"
            req = urllib.request.Request(self.superhead_url, data=json_canon(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            ch = (obj.get("choices") or [{}])[0]
            msg = (ch.get("message") or {})
            return (msg.get("content") or "").strip() or None
        except Exception:
            return None

    def chat(self, message: str, session_id: str = "") -> Dict[str, Any]:
        msg = (message or "").strip()
        if not msg:
            return {"ok": False, "error": "empty message"}

        hits = self.index.search(msg, k=5)
        ctx = ""
        if hits:
            parts = []
            for h in hits[:3]:
                parts.append(f"[{h['doc_id']}] {h['title']} ({h['source']})\n{h['snippet'][:800]}")
            ctx = "\n\n---\n\n".join(parts)

        # optional superhead
        sys = "You are OCTA Fusion Agent. Be concise, operational, and cite doc_ids when using corpus context."
        messages = [{"role": "system", "content": sys}]
        if ctx:
            messages.append({"role": "system", "content": f"Corpus context:\n{ctx}"})
        messages.append({"role": "user", "content": msg})

        out = self._call_superhead(messages)
        if not out:
            # deterministic fallback response
            if hits:
                out = (
                    "I searched your local corpus and found relevant material:\n\n"
                    + "\n".join([f"- {h['doc_id']} · {h['title']} (score={h['score']:.3f})" for h in hits[:5]])
                    + "\n\nAsk a more specific question, or request: `summarize doc <id>`."
                )
            else:
                out = "No local corpus hits yet. Ingest text/URLs/files, or enable curriculum runner to populate the index."

        # stimulate nodeA (light touch) based on message hash
        vec = text_to_stim_vec(msg, n=4)
        self.stimulate(region="Chat", vec=vec, target="nodeA")

        self.bus.emit({
            "type": "chat",
            "tag": "[chat]",
            "text": out[:1800],
            "metrics": {"session_id": session_id, "hits": len(hits)},
            "witness": "machine · agent",
            "ts_ms": now_ms(),
        })

        return {"ok": True, "reply": out, "hits": hits}

# ======================================================================================
# CLM BRIDGE (Agent ingest -> deterministic CLM event vectors)
# ======================================================================================

def _text_to_clm_vec(text: str, dim: int) -> List[float]:
    """
    Deterministic vectorization from text hash.
    Not semantic; designed for auditable linkage into MA log and router.
    """
    base = hashlib.sha256((text or "").encode("utf-8")).digest()
    out: List[float] = []
    ctr = 0
    while len(out) < dim:
        h = hashlib.sha256(base + ctr.to_bytes(4, "big")).digest()
        for i in range(0, len(h), 4):
            if len(out) >= dim:
                break
            u = int.from_bytes(h[i:i+4], "big")
            # map uint32 -> [-1, 1]
            x = (u / 0xFFFFFFFF) * 2.0 - 1.0
            out.append(float(x))
        ctr += 1
    return out

class CLMBridge:
    def __init__(self, clm_store: CLMStore, clm_kernel: CLMKernel, dim_raw: int, enabled: bool = True):
        self.store = clm_store
        self.kernel = clm_kernel
        self.dim_raw = int(dim_raw)
        self.enabled = bool(enabled)

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "dim_raw": self.dim_raw}

    def ingest_text(self, doc_id: str, title: str, source: str, source_key: str, text: str):
        if not self.enabled:
            return
        # deterministic event_id from content hash (doc_id already derived; still include source_key for uniqueness)
        eid = sha256_hex(json_canon({"doc_id": doc_id, "source_key": source_key}).encode("utf-8"))[:24]
        x = _text_to_clm_vec(text, self.dim_raw)
        meta = {
            "bridge": "agent->clm",
            "doc_id": doc_id,
            "title": title[:240],
            "source": source[:120],
            "source_key": source_key[:512],
            "ts_ms": now_ms(),
        }
        self.store.put_event(eid, x, meta)
        self.kernel.ma.trace({"type": "bridge_ingest", "event_id": eid, "doc_id": doc_id, "source": source})

# ======================================================================================
# HTTP SERVER (Fusion): CLM endpoints + Agent endpoints (SSE)
# ======================================================================================

class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

class FusionHTTP:
    def __init__(self, host: str, port: int, clm_kernel: CLMKernel, clm_store: CLMStore,
                 agent: Optional[ClusterAgent], admin_key: str):
        self.host = host
        self.port = int(port)
        self.clm_kernel = clm_kernel
        self.clm_store = clm_store
        self.agent = agent
        self.admin_key = admin_key

        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            server_version = "OCTA-Fusion/1.0"

            def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, code: int, obj: Any):
                self._send(code, json_canon(obj).encode("utf-8"))

            def _read_json(self) -> Dict[str, Any]:
                n = int(self.headers.get("Content-Length", "0") or "0")
                if n <= 0 or n > 5_000_000:
                    return {}
                raw = self.rfile.read(n)
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return {}

            def _path(self) -> str:
                return urllib.parse.urlparse(self.path).path

            def _q(self) -> Dict[str, List[str]]:
                return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

            # ---------------- CLM ROUTES ----------------

            def _clm_health(self):
                self._send_json(200, {"ok": True, "service": "clm", "ts_utc": utc_now_iso()})

            def _clm_metrics(self):
                # light metrics only (avoid heavy scans)
                with parent.clm_store._lock:
                    ev = parent.clm_store.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
                    lb = parent.clm_store.conn.execute("SELECT COUNT(*) AS c FROM labels").fetchone()["c"]
                    ops = parent.clm_store.conn.execute("SELECT COUNT(*) AS c FROM ma_ops").fetchone()["c"]
                    models = parent.clm_store.conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()["c"]
                self._send_json(200, {
                    "ok": True,
                    "events": int(ev),
                    "labels": int(lb),
                    "ma_ops": int(ops),
                    "models": int(models),
                    "ma_tail": parent.clm_store.ma_tail_hash(),
                    "cursor_label_id": parent.clm_kernel.cursor_label_id,
                    "canary_pct": parent.clm_kernel.canary_pct,
                    "ts_utc": utc_now_iso(),
                })

            def _clm_model_current(self):
                mid = parent.clm_store.get_deployment("current")
                if not mid:
                    self._send_json(200, {"ok": True, "model_id": None})
                    return
                row = parent.clm_store.get_model(mid)
                self._send_json(200, {"ok": True, "model_id": mid, "model": row})

            def _clm_ingest(self):
                obj = self._read_json()
                event_id = str(obj.get("event_id", "")).strip()
                x = obj.get("x", None)
                meta = obj.get("meta", {}) or {}

                if not event_id or not isinstance(x, list):
                    self._send_json(400, {"ok": False, "error": "expected {event_id,x,meta}"})
                    return
                if len(x) != parent.clm_kernel.cfg.dim_raw:
                    self._send_json(400, {"ok": False, "error": f"dim mismatch: expected {parent.clm_kernel.cfg.dim_raw}"})
                    return

                parent.clm_store.put_event(event_id, [float(v) for v in x], dict(meta))
                h = parent.clm_kernel.ma.trace({"type": "event_ingest", "event_id": event_id, "meta": meta})
                self._send_json(200, {"ok": True, "event_id": event_id, "ma_op": h})

            def _clm_label(self):
                obj = self._read_json()
                event_id = str(obj.get("event_id", "")).strip()
                y = obj.get("y", None)
                source = str(obj.get("source", "unknown"))[:80]
                if not event_id or y is None:
                    self._send_json(400, {"ok": False, "error": "expected {event_id,y,source}"})
                    return
                if int(y) not in (0, 1):
                    self._send_json(400, {"ok": False, "error": "y must be 0|1"})
                    return
                if not parent.clm_store.get_event(event_id):
                    self._send_json(404, {"ok": False, "error": "event_id not found"})
                    return
                ver = parent.clm_store.put_label(event_id, int(y), source)
                parent.clm_kernel.ma.trace({"type": "label", "event_id": event_id, "y": int(y), "source": source, "version": ver})
                self._send_json(200, {"ok": True, "event_id": event_id, "version": ver})

            def _clm_train_tick(self):
                obj = self._read_json()
                max_cycles = int(obj.get("max_cycles", 1))
                try:
                    out = parent.clm_kernel.train_tick(max_cycles=max(1, min(50, max_cycles)))
                    self._send_json(200, {"ok": True, **out})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": str(e)})

            def _clm_predict(self):
                obj = self._read_json()
                x = obj.get("x", None)
                user = str(obj.get("api_key", "") or obj.get("user", "")).strip()
                request_id = str(obj.get("request_id", "")).strip() or sha256_hex(json_canon({"t": now_ms(), "r": random.random()}).encode("utf-8"))[:24]
                if not isinstance(x, list) or not user:
                    self._send_json(400, {"ok": False, "error": "expected {x, api_key/user, request_id?}"})
                    return

                if not parent.clm_store.credits_charge(user, parent.clm_kernel.cfg.cost_per_predict, "predict", {"request_id": request_id}):
                    self._send_json(402, {"ok": False, "error": "insufficient credits"})
                    return

                try:
                    pred = parent.clm_kernel.predict([float(v) for v in x], request_id=request_id)
                    self._send_json(200, {"ok": True, "request_id": request_id, "charged": parent.clm_kernel.cfg.cost_per_predict, **pred})
                except Exception as e:
                    # refund on failure
                    parent.clm_store.credits_issue(user, parent.clm_kernel.cfg.cost_per_predict, "refund_predict_error", {"request_id": request_id, "err": str(e)})
                    self._send_json(500, {"ok": False, "error": str(e)})

            def _clm_deploy_canary(self):
                obj = self._read_json()
                admin = str(obj.get("admin_key", "")).strip()
                if admin != parent.admin_key:
                    self._send_json(403, {"ok": False, "error": "forbidden"})
                    return
                cand = obj.get("candidate_model_id", None)
                pct = int(obj.get("canary_pct", 0))
                try:
                    parent.clm_kernel.set_canary(candidate_model_id=(str(cand).strip() if cand else None), canary_pct=pct)
                    self._send_json(200, {"ok": True, "candidate_model_id": parent.clm_kernel.candidate_model_id, "canary_pct": parent.clm_kernel.canary_pct})
                except Exception as e:
                    self._send_json(400, {"ok": False, "error": str(e)})

            def _clm_credits_issue(self):
                obj = self._read_json()
                admin = str(obj.get("admin_key", "")).strip()
                if admin != parent.admin_key:
                    self._send_json(403, {"ok": False, "error": "forbidden"})
                    return
                user = str(obj.get("user", "")).strip()
                credits = int(obj.get("credits", 0))
                if not user or credits == 0:
                    self._send_json(400, {"ok": False, "error": "expected {user,credits}"})
                    return
                parent.clm_store.credits_issue(user, credits, "issue", {"admin": True})
                self._send_json(200, {"ok": True, "user": user, "balance": parent.clm_store.credits_balance(user)})

            def _clm_credits_balance(self):
                obj = self._read_json()
                user = str(obj.get("user", "")).strip()
                if not user:
                    self._send_json(400, {"ok": False, "error": "expected {user}"})
                    return
                self._send_json(200, {"ok": True, "user": user, "balance": parent.clm_store.credits_balance(user)})

            # ---- Sync (MA ops + models + deployments) ----

            def _sync_status(self):
                dep = {
                    "current": parent.clm_store.get_deployment("current"),
                    "candidate": parent.clm_store.get_deployment("candidate"),
                }
                self._send_json(200, {"ok": True, "ma_tail": parent.clm_store.ma_tail_hash(), "deployments": dep, "canary_pct": parent.clm_kernel.canary_pct})

            def _sync_pull(self):
                obj = self._read_json()
                from_hash = str(obj.get("from_hash", "0"*64))
                limit = int(obj.get("limit", 500))
                limit = max(1, min(5000, limit))

                ops = parent.clm_store.ma_pull_since(from_hash, limit)

                dep = {
                    "current": parent.clm_store.get_deployment("current"),
                    "candidate": parent.clm_store.get_deployment("candidate"),
                }
                models: List[Dict[str, Any]] = []
                # include deployed models (if available)
                for mid in set([v for v in dep.values() if v]):
                    row = parent.clm_store.get_model(mid)
                    if row:
                        models.append({
                            "model_id": row["model_id"],
                            "ts_utc": row["ts_utc"],
                            "algo": row["algo"],
                            "dim": int(row["dim"]),
                            "blob_b64": row["blob_b64"],
                            "metrics_json": row["metrics_json"],
                            "config_json": row["config_json"],
                            "data_fingerprint": row["data_fingerprint"],
                        })

                self._send_json(200, {
                    "ok": True,
                    "ops": ops,
                    "deployments": dep,
                    "canary_pct": parent.clm_kernel.canary_pct,
                    "models": models,
                    "ma_tail": parent.clm_store.ma_tail_hash(),
                })

            def _sync_push(self):
                obj = self._read_json()
                ops = obj.get("ops", []) or []
                models = obj.get("models", []) or []
                imported_ops = 0
                imported_models = 0

                # import models first
                for m in models:
                    try:
                        mid = str(m.get("model_id", "")).strip()
                        if not mid:
                            continue
                        if parent.clm_store.get_model(mid):
                            continue
                        blob_b64 = str(m.get("blob_b64", "")).strip()
                        blob = base64.b64decode(blob_b64.encode("ascii"))
                        metrics = json.loads(str(m.get("metrics_json", "{}")))
                        config = json.loads(str(m.get("config_json", "{}")))
                        algo = str(m.get("algo", "unknown"))
                        dim = int(m.get("dim", parent.clm_kernel.cfg.orth_k))
                        fp = str(m.get("data_fingerprint", ""))
                        parent.clm_store.put_model(mid, algo=algo, dim=dim, blob=blob, metrics=metrics, config=config, data_fingerprint=fp)
                        imported_models += 1
                    except Exception:
                        continue

                # import ops
                for o in ops:
                    try:
                        ok = parent.clm_store.ma_import_op(o["prev_hash"], o["op"], o["op_hash"])
                        if ok:
                            imported_ops += 1
                    except Exception:
                        continue

                self._send_json(200, {"ok": True, "imported_ops": imported_ops, "imported_models": imported_models, "ma_tail": parent.clm_store.ma_tail_hash()})

            # ---------------- AGENT ROUTES ----------------

            def _agent_require(self) -> bool:
                if parent.agent is None:
                    self._send_json(404, {"ok": False, "error": "agent disabled"})
                    return False
                return True

            def _agent_status(self):
                if not self._agent_require():
                    return
                self._send_json(200, {"ok": True, **parent.agent.status()})

            def _agent_events_sse(self):
                if not self._agent_require():
                    return
                # SSE headers
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # stream loop
                import queue
                q = parent.agent.bus.q
                self.wfile.write(b": ok\n\n")
                self.wfile.flush()

                while True:
                    try:
                        ev = q.get(timeout=1.0)
                        payload = json_canon(ev).encode("utf-8")
                        self.wfile.write(b"event: ev\n")
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                    except queue.Empty:
                        # keepalive
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                    except Exception:
                        break

            def _agent_chat(self):
                if not self._agent_require():
                    return
                obj = self._read_json()
                msg = str(obj.get("message", "") or "").strip()
                sid = str(obj.get("session_id", "") or "").strip()
                out = parent.agent.chat(msg, session_id=sid)
                self._send_json(200, out)

            def _agent_ingest(self):
                if not self._agent_require():
                    return
                obj = self._read_json()
                kind = str(obj.get("kind", "")).strip().lower()
                if kind not in ("text", "url", "file", "folder"):
                    self._send_json(400, {"ok": False, "error": "kind must be text|url|file|folder"})
                    return

                job = {"kind": kind}
                for k in ("text", "url", "path", "title", "source", "source_key"):
                    if k in obj:
                        job[k] = obj[k]
                parent.agent.enqueue_ingest(job)
                self._send_json(200, {"ok": True, "enqueued": True, "kind": kind})

            def _agent_corpus_search(self):
                if not self._agent_require():
                    return
                q = (self._q().get("q", [""]) or [""])[0]
                hits = parent.agent.index.search(q, k=6)
                self._send_json(200, {"ok": True, "q": q, "hits": hits})

            def _agent_corpus_doc(self):
                if not self._agent_require():
                    return
                q = self._q()
                did = (q.get("id", [""]) or [""])[0].strip()
                d = parent.agent.index.get_doc(did)
                if not d:
                    self._send_json(404, {"ok": False, "error": "doc not found"})
                    return
                self._send_json(200, {"ok": True, "doc": d})

            def _agent_set_rate(self):
                if not self._agent_require():
                    return
                obj = self._read_json()
                hz = float(obj.get("hz", 10.0))
                parent.agent.set_rate_all(hz)
                self._send_json(200, {"ok": True, "hz": hz})

            def _agent_stimulate(self):
                if not self._agent_require():
                    return
                obj = self._read_json()
                if "text" in obj and obj["text"]:
                    vec = text_to_stim_vec(str(obj["text"]), n=4)
                    target = str(obj.get("target", "nodeA"))
                    region = str(obj.get("region", "Text"))
                    parent.agent.stimulate(region=region, vec=vec, target=target)
                    self._send_json(200, {"ok": True, "target": target, "region": region, "vec": vec})
                    return

                target = str(obj.get("target", "nodeA"))
                region = str(obj.get("region", "Region_X"))
                vec = obj.get("vec", [])
                if not isinstance(vec, list) or len(vec) < 1:
                    self._send_json(400, {"ok": False, "error": "expected {target,region,vec} or {text}"})
                    return
                parent.agent.stimulate(region=region, vec=[float(v) for v in vec], target=target)
                self._send_json(200, {"ok": True, "target": target, "region": region, "vec": vec})

            def _agent_ledger_verify(self):
                if not self._agent_require():
                    return
                self._send_json(200, {"ok": True, "verify": parent.agent.ledger_verify()})

            def _agent_curriculum_start(self):
                if not self._agent_require():
                    return
                parent.agent.curriculum.enable(True)
                self._send_json(200, {"ok": True, "enabled": True, "status": parent.agent.curriculum.status()})

            def _agent_curriculum_stop(self):
                if not self._agent_require():
                    return
                parent.agent.curriculum.enable(False)
                self._send_json(200, {"ok": True, "enabled": False, "status": parent.agent.curriculum.status()})

            def _agent_curriculum_status(self):
                if not self._agent_require():
                    return
                self._send_json(200, {"ok": True, "status": parent.agent.curriculum.status()})

            def _agent_export_actor_c(self):
                if not self._agent_require():
                    return
                obj = self._read_json()
                node = str(obj.get("node", "nodeA"))
                actor_index = int(obj.get("actor_index", 0))
                out_path = str(obj.get("out_path", "./actor0.h"))
                scale = float(obj.get("scale", 64))
                parent.agent.export_actor_c(node=node, actor_index=actor_index, out_path=out_path, scale=scale)
                self._send_json(200, {"ok": True, "out_path": out_path})

            # ---------------- Router ----------------

            def do_GET(self):
                p = self._path()

                # agent routes
                if p.startswith("/api/"):
                    if p == "/api/status":
                        return self._agent_status()
                    if p == "/api/events":
                        return self._agent_events_sse()
                    if p == "/api/corpus/search":
                        return self._agent_corpus_search()
                    if p == "/api/corpus/doc":
                        return self._agent_corpus_doc()
                    if p == "/api/ledger_verify":
                        return self._agent_ledger_verify()
                    if p == "/api/curriculum/status":
                        return self._agent_curriculum_status()
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return

                # clm routes
                if p == "/health":
                    return self._clm_health()
                if p == "/metrics":
                    return self._clm_metrics()
                if p == "/model_current":
                    return self._clm_model_current()
                if p == "/sync/status":
                    return self._sync_status()

                self._send_json(404, {"ok": False, "error": "not found"})

            def do_POST(self):
                p = self._path()

                # agent routes
                if p.startswith("/api/"):
                    if p == "/api/chat":
                        return self._agent_chat()
                    if p == "/api/ingest":
                        return self._agent_ingest()
                    if p == "/api/set_rate":
                        return self._agent_set_rate()
                    if p == "/api/stimulate":
                        return self._agent_stimulate()
                    if p == "/api/curriculum/start":
                        return self._agent_curriculum_start()
                    if p == "/api/curriculum/stop":
                        return self._agent_curriculum_stop()
                    if p == "/api/export_actor_c":
                        return self._agent_export_actor_c()
                    self._send_json(404, {"ok": False, "error": "not found"})
                    return

                # clm routes
                if p == "/ingest":
                    return self._clm_ingest()
                if p == "/label":
                    return self._clm_label()
                if p == "/train_tick":
                    return self._clm_train_tick()
                if p == "/predict":
                    return self._clm_predict()
                if p == "/deploy_canary":
                    return self._clm_deploy_canary()
                if p == "/credits_issue":
                    return self._clm_credits_issue()
                if p == "/credits_balance":
                    return self._clm_credits_balance()
                if p == "/sync/pull":
                    return self._sync_pull()
                if p == "/sync/push":
                    return self._sync_push()

                self._send_json(404, {"ok": False, "error": "not found"})

            def log_message(self, fmt: str, *args):
                LOG.info("%s - %s" % (self.address_string(), fmt % args))

        self._httpd = _ThreadingHTTPServer((self.host, self.port), Handler)

    def serve_forever(self):
        LOG.info(f"[fusion] http server listening on http://{self.host}:{self.port}")
        self._httpd.serve_forever()

# ======================================================================================
# CLI: CLM / CLUSTER / FUSION
# ======================================================================================

def clm_init_cmd(args: argparse.Namespace) -> int:
    store = CLMStore(args.db)
    store.init()
    store.close()
    print(json_canon({"ok": True, "db": args.db}))
    return 0

def clm_simulate_cmd(args: argparse.Namespace) -> int:
    store = CLMStore(args.db)
    store.init()
    cfg = CLMConfig(dim_raw=int(args.dim))
    cfg.admin_key = args.admin_key or cfg.admin_key
    kernel = CLMKernel(store, cfg)

    rng = np.random.default_rng(cfg.seed)
    n = int(args.n)
    dim = int(args.dim)

    # synthetic: separable by a hidden vector
    w = rng.normal(size=(dim,))
    w = w / (np.linalg.norm(w) + 1e-12)

    # create events + labels
    for i in range(n):
        x = rng.normal(size=(dim,)).astype(np.float64)
        p = float(sigmoid(np.array([x @ w]))[0])
        y = 1 if rng.random() < p else 0
        eid = sha256_hex(json_canon({"i": i, "salt": "sim"}).encode("utf-8"))[:24]
        store.put_event(eid, x.tolist(), {"sim": True, "i": i})
        store.put_label(eid, y, "sim")

    out = kernel.train_tick(max_cycles=max(1, int(args.cycles)))
    print(json_canon({"ok": True, **out}))
    store.close()
    return 0

def clm_serve_cmd(args: argparse.Namespace) -> int:
    store = CLMStore(args.db)
    store.init()
    cfg = CLMConfig(dim_raw=int(args.dim))
    cfg.admin_key = args.admin_key or cfg.admin_key
    kernel = CLMKernel(store, cfg)

    peers = [p.strip() for p in (args.peers or "").split(",") if p.strip()]
    gossip = None
    if peers:
        gossip = CLMGossipLoop(store=store, kernel=kernel, peers=peers, interval_s=cfg.gossip_interval_s, batch_limit=cfg.sync_batch_limit)
        gossip.start()

    http = FusionHTTP(host=args.host, port=int(args.port), clm_kernel=kernel, clm_store=store, agent=None, admin_key=cfg.admin_key)
    try:
        http.serve_forever()
    finally:
        if gossip:
            gossip.stop()
        store.close()
    return 0

async def _cluster_run_async(args: argparse.Namespace) -> None:
    host = "127.0.0.1"
    base = int(args.mesh_base)
    data_dir = args.data_dir

    nodeA = Node("nodeA", host, base + 0, peers=[(host, base + 1), (host, base + 2)], data_dir=data_dir, seed=17)
    nodeB = Node("nodeB", host, base + 1, peers=[(host, base + 0), (host, base + 2)], data_dir=data_dir, seed=19)
    nodeC = Node("nodeC", host, base + 2, peers=[(host, base + 0), (host, base + 1)], data_dir=data_dir, seed=23)

    await nodeA.start()
    await nodeB.start()
    await nodeC.start()

    nodes = {"nodeA": nodeA, "nodeB": nodeB, "nodeC": nodeC}
    agent = ClusterAgent(nodes=nodes, data_dir=data_dir, clm_bridge=None)

    # standalone agent HTTP (non-fusion)
    clm_store = CLMStore(":memory:")
    clm_store.init()
    cfg = CLMConfig(dim_raw=64)
    kernel = CLMKernel(clm_store, cfg)

    http = FusionHTTP(host=args.agent_host, port=int(args.agent_port), clm_kernel=kernel, clm_store=clm_store, agent=agent, admin_key="CHANGE_ME")
    th = threading.Thread(target=http.serve_forever, daemon=True)
    th.start()

    LOG.info(f"[cluster] mesh on {base}..{base+2} · agent http http://{args.agent_host}:{args.agent_port}")
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            agent.stop()
        except Exception:
            pass
        await nodeA.stop()
        await nodeB.stop()
        await nodeC.stop()

def cluster_run_cmd(args: argparse.Namespace) -> int:
    asyncio.run(_cluster_run_async(args))
    return 0

async def _fusion_run_async(args: argparse.Namespace) -> None:
    store = CLMStore(args.db)
    store.init()
    cfg = CLMConfig(dim_raw=int(args.dim))
    cfg.admin_key = args.admin_key or cfg.admin_key
    kernel = CLMKernel(store, cfg)

    peers = [p.strip() for p in (args.peers or "").split(",") if p.strip()]
    gossip = None
    if peers:
        gossip = CLMGossipLoop(store=store, kernel=kernel, peers=peers, interval_s=cfg.gossip_interval_s, batch_limit=cfg.sync_batch_limit)
        gossip.start()

    agent = None
    if args.agent_enable:
        host = "127.0.0.1"
        base = int(args.mesh_base)
        data_dir = args.data_dir

        nodeA = Node("nodeA", host, base + 0, peers=[(host, base + 1), (host, base + 2)], data_dir=data_dir, seed=17)
        nodeB = Node("nodeB", host, base + 1, peers=[(host, base + 0), (host, base + 2)], data_dir=data_dir, seed=19)
        nodeC = Node("nodeC", host, base + 2, peers=[(host, base + 0), (host, base + 1)], data_dir=data_dir, seed=23)

        await nodeA.start()
        await nodeB.start()
        await nodeC.start()

        bridge = CLMBridge(clm_store=store, clm_kernel=kernel, dim_raw=cfg.dim_raw, enabled=True)
        nodes = {"nodeA": nodeA, "nodeB": nodeB, "nodeC": nodeC}
        agent = ClusterAgent(nodes=nodes, data_dir=data_dir, clm_bridge=bridge)

    http = FusionHTTP(host=args.host, port=int(args.port), clm_kernel=kernel, clm_store=store, agent=agent, admin_key=cfg.admin_key)
    try:
        http.serve_forever()
    finally:
        if gossip:
            gossip.stop()
        if agent:
            try:
                agent.stop()
            except Exception:
                pass
        store.close()

def fusion_serve_cmd(args: argparse.Namespace) -> int:
    asyncio.run(_fusion_run_async(args))
    return 0

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="octa_fusion_v1.py")
    p.add_argument("--log", default="INFO", help="log level (DEBUG,INFO,WARNING,ERROR)")
    sub = p.add_subparsers(dest="cmd")

    # CLM
    clm = sub.add_parser("clm", help="CLM kernel commands")
    clm_sub = clm.add_subparsers(dest="clm_cmd")

    clm_init = clm_sub.add_parser("init", help="init sqlite schema")
    clm_init.add_argument("--db", required=True)
    clm_init.set_defaults(func=clm_init_cmd)

    clm_sim = clm_sub.add_parser("simulate", help="synthetic data simulation + train")
    clm_sim.add_argument("--db", required=True)
    clm_sim.add_argument("--dim", type=int, default=64)
    clm_sim.add_argument("--n", type=int, default=20000)
    clm_sim.add_argument("--cycles", type=int, default=3)
    clm_sim.add_argument("--admin-key", default="CHANGE_ME_ADMIN_KEY")
    clm_sim.set_defaults(func=clm_simulate_cmd)

    clm_srv = clm_sub.add_parser("serve", help="serve CLM API")
    clm_srv.add_argument("--db", required=True)
    clm_srv.add_argument("--dim", type=int, default=64)
    clm_srv.add_argument("--host", default="0.0.0.0")
    clm_srv.add_argument("--port", type=int, default=8080)
    clm_srv.add_argument("--admin-key", default="CHANGE_ME_ADMIN_KEY")
    clm_srv.add_argument("--peers", default="", help="comma-separated peer base urls (for gossip)")
    clm_srv.set_defaults(func=clm_serve_cmd)

    # CLUSTER
    cluster = sub.add_parser("cluster", help="P3P cluster + agent commands")
    cluster_sub = cluster.add_subparsers(dest="cluster_cmd")

    cluster_run = cluster_sub.add_parser("run", help="run 3-node localhost mesh + standalone agent http")
    cluster_run.add_argument("--mesh-base", type=int, default=10001)
    cluster_run.add_argument("--data-dir", default="./data_fusion")
    cluster_run.add_argument("--agent-host", default="0.0.0.0")
    cluster_run.add_argument("--agent-port", type=int, default=9090)
    cluster_run.set_defaults(func=cluster_run_cmd)

    # FUSION
    fusion = sub.add_parser("fusion", help="single-port fusion server (CLM + Agent)")
    fusion_sub = fusion.add_subparsers(dest="fusion_cmd")

    fusion_srv = fusion_sub.add_parser("serve", help="serve fusion http on one port")
    fusion_srv.add_argument("--db", required=True)
    fusion_srv.add_argument("--dim", type=int, default=64)
    fusion_srv.add_argument("--host", default="0.0.0.0")
    fusion_srv.add_argument("--port", type=int, default=8080)
    fusion_srv.add_argument("--admin-key", default="CHANGE_ME_ADMIN_KEY")
    fusion_srv.add_argument("--peers", default="", help="comma-separated peer base urls for CLM gossip")
    fusion_srv.add_argument("--agent-enable", action="store_true", help="enable 3-node mesh + agent endpoints")
    fusion_srv.add_argument("--mesh-base", type=int, default=10001)
    fusion_srv.add_argument("--data-dir", default="./data_fusion")
    fusion_srv.set_defaults(func=fusion_serve_cmd)

    return p

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log)

    if not args.cmd:
        parser.print_help()
        return 2

    # normalize nested subcommands
    if args.cmd == "clm" and not getattr(args, "clm_cmd", None):
        parser.print_help()
        return 2
    if args.cmd == "cluster" and not getattr(args, "cluster_cmd", None):
        parser.print_help()
        return 2
    if args.cmd == "fusion" and not getattr(args, "fusion_cmd", None):
        parser.print_help()
        return 2

    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
