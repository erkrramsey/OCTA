#!/usr/bin/env python3
# ======================================================================================
# NUMCHAIN · v1.4.2 (v1.4.1 + TX GOSSIP + ACK/RETRY RELIABILITY)
# Single-file kernel: deterministic genesis + MA state engine + minimal BFT + TCP gossip +
# FastAPI RPC + flood/poll tools.
#
# What changed vs v1.4.1:
#   1) TX propagation: /tx accepted → gossiped; TXGOSSIP received → validate/dedup → re-gossip (TTL)
#   2) Reliability: consensus-critical gossip messages use ACK + retry with backpressure
#        critical: PROPOSAL, VOTE, QC, COMMIT, RANDAO_COMMIT, RANDAO_REVEAL
#
# What is NOT added (by design):
#   - No libp2p, no NAT traversal, no real WAN hardening
#   - No persistent DB (append-only JSONL log only)
#   - No validator set changes
#
# Quickstart (local):
#   python numchain_v142.py deps
#   python numchain_v142.py genesis --out genesis.json --nodes 5
#   # Terminal A:
#   python numchain_v142.py run --genesis genesis.json --node-index 0 --rpc 8000 --gossip 9000 --peers 127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004
#   # Terminal B..E (node-index 1..4; rpc 8001..8004; gossip 9001..9004; peers list all others)
#
# Flood (hits one node, now stresses all via tx gossip):
#   python numchain_v142.py flood --url http://127.0.0.1:8000 --seconds 60 --target-tps 4000 --senders 512 --ramp-up 10
#
# Poll:
#   python numchain_v142.py poll --urls http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004 --seconds 60 --every 2
#
# ======================================================================================

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import hmac
import json
import math
import os
import queue
import random
import socket
import statistics
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Optional deps (install via `deps`)
# ----------------------------
try:
    import numpy as np
except Exception:
    np = None

try:
    import psutil
except Exception:
    psutil = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
except Exception:
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    PlainTextResponse = None  # type: ignore

try:
    import uvicorn
except Exception:
    uvicorn = None  # type: ignore

try:
    import httpx
except Exception:
    httpx = None  # type: ignore

# Ed25519: prefer PyNaCl; fallback to cryptography if available.
ED_IMPL = "none"
try:
    from nacl.signing import SigningKey, VerifyKey  # type: ignore

    ED_IMPL = "pynacl"
except Exception:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey  # type: ignore

        ED_IMPL = "cryptography"
    except Exception:
        ED_IMPL = "none"

VERSION = "v1.4.2"
CHAIN_ID_DEFAULT = "numchain-dev"
GENESIS_VERSION = 1

# ======================================================================================
# Helpers: canonical JSON, hashing, time, stats
# ======================================================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def jcanon(obj: Any) -> bytes:
    # Deterministic serialization: sorted keys, no whitespace.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def hhex(b: bytes, n: int = 12) -> str:
    return b.hex()[:n]

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def quantize(x: float, q: int) -> int:
    # round-half-away-from-zero deterministic-ish:
    v = x * q
    if v >= 0:
        return int(v + 0.5)
    return -int((-v) + 0.5)

def dequantize(i: int, q: int) -> float:
    return float(i) / float(q)

# ======================================================================================
# Ed25519 signing
# ======================================================================================

class Ed25519:
    def __init__(self, sk_bytes: Optional[bytes] = None):
        if ED_IMPL == "none":
            raise RuntimeError("Ed25519 unavailable. Run: python numchain_v142.py deps")
        if sk_bytes is None:
            sk_bytes = os.urandom(32)
        self._sk_bytes = sk_bytes
        if ED_IMPL == "pynacl":
            self._sk = SigningKey(sk_bytes)
            self._vk = self._sk.verify_key
        else:
            self._sk = Ed25519PrivateKey.from_private_bytes(sk_bytes)
            self._vk = self._sk.public_key()

    @property
    def privkey_b64(self) -> str:
        return b64e(self._sk_bytes)

    def pubkey_bytes(self) -> bytes:
        if ED_IMPL == "pynacl":
            return bytes(self._vk)
        return self._vk.public_bytes()  # type: ignore[attr-defined]

    def pubkey_b64(self) -> str:
        return b64e(self.pubkey_bytes())

    def sign(self, msg: bytes) -> bytes:
        if ED_IMPL == "pynacl":
            return self._sk.sign(msg).signature
        return self._sk.sign(msg)  # type: ignore[no-any-return]

    @staticmethod
    def verify(pubkey_bytes: bytes, msg: bytes, sig: bytes) -> bool:
        try:
            if ED_IMPL == "pynacl":
                VerifyKey(pubkey_bytes).verify(msg, sig)
                return True
            Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(sig, msg)  # type: ignore
            return True
        except Exception:
            return False

# ======================================================================================
# Simple bounded LRU set (dedup)
# ======================================================================================

class LRUSet:
    def __init__(self, cap: int):
        self.cap = int(cap)
        self._d: Dict[str, int] = {}
        self._q: List[str] = []
        self._tick = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._d

    def add(self, key: str) -> None:
        with self._lock:
            self._tick += 1
            if key in self._d:
                self._d[key] = self._tick
                return
            self._d[key] = self._tick
            self._q.append(key)
            if len(self._d) > self.cap:
                # evict oldest approx: pop until find stale entry
                while self._q and len(self._d) > self.cap:
                    k = self._q.pop(0)
                    if k in self._d and self._d[k] <= self._tick - self.cap:
                        del self._d[k]

# ======================================================================================
# MA fixed-point memory arithmetic (per-shard)
# ======================================================================================

@dataclass
class MAParams:
    L: int = 4       # layers
    S: int = 64      # slots
    D: int = 128     # dims
    decay: float = 0.985
    tau: float = 1.5
    clip: float = 8.0
    q: int = 1024    # fixed-point scale

def ma_init(params: MAParams, seed: int) -> "np.ndarray":
    assert np is not None
    rng = np.random.default_rng(seed)
    # fixed-point int32 tensor [L,S,D]
    x = rng.normal(0, 0.05, size=(params.L, params.S, params.D)).astype(np.float32)
    x = np.clip(x, -params.clip, params.clip)
    xi = np.round(x * params.q).astype(np.int32)
    return xi

def ma_apply_tx(state: "np.ndarray", params: MAParams, tx: Dict[str, Any]) -> None:
    """
    Bounded ops; deterministic transforms.
    tx fields used:
      shard, layer, slot, op, vec_b64, scalar, decay_override(optional)
    """
    shard = int(tx["shard"])
    layer = int(tx["layer"])
    slot = int(tx["slot"])
    op = tx["op"]

    # State is per-shard owned outside; here it's shard-local tensor.
    # state shape [L,S,D]
    if op == "add":
        vb = b64d(tx["vec_b64"])
        v = np.frombuffer(vb, dtype=np.int16).astype(np.int32)  # int16 payload
        if v.shape[0] != params.D:
            return
        # clamp delta norm
        dn = int(tx.get("delta_norm_max", params.q * 2))
        # quick norm bound (L1) deterministic
        l1 = int(np.sum(np.abs(v)))
        if l1 > dn:
            scale = dn / float(l1)
            v = np.round(v.astype(np.float64) * scale).astype(np.int32)
        state[layer, slot, :] = np.clip(state[layer, slot, :] + v, -params.clip * params.q, params.clip * params.q).astype(np.int32)

    elif op == "decay":
        d = float(tx.get("decay", params.decay))
        d = clamp(d, 0.8, 0.9999)
        # fixed-point multiply with rounding
        state[layer, slot, :] = np.round(state[layer, slot, :].astype(np.float64) * d).astype(np.int32)

    elif op == "mix":
        # mix slot with another slot
        src = int(tx.get("src_slot", 0))
        alpha = float(tx.get("alpha", 0.5))
        alpha = clamp(alpha, 0.0, 1.0)
        a = alpha
        b = 1.0 - a
        mixed = np.round(state[layer, slot, :].astype(np.float64) * a + state[layer, src, :].astype(np.float64) * b).astype(np.int32)
        state[layer, slot, :] = np.clip(mixed, -params.clip * params.q, params.clip * params.q).astype(np.int32)

    elif op == "clip":
        # explicit clamp
        state[layer, slot, :] = np.clip(state[layer, slot, :], -params.clip * params.q, params.clip * params.q).astype(np.int32)

# ======================================================================================
# Protocol objects: tx, blocks, receipts
# ======================================================================================

def tx_canonical_message(genesis_hash: str, tx: Dict[str, Any]) -> bytes:
    # Tx message excludes signature; include genesis hash for domain separation.
    msg = {"genesis": genesis_hash, "tx": tx}
    return jcanon(msg)

def tx_id(genesis_hash: str, tx: Dict[str, Any]) -> str:
    return sha256(tx_canonical_message(genesis_hash, tx)).hex()

def env_id(env: Dict[str, Any]) -> str:
    # deterministic id of envelope without signature (still stable)
    e = dict(env)
    e.pop("sig_b64", None)
    return sha256(jcanon(e)).hex()

# ======================================================================================
# TCP gossip framing: length-prefixed JSON
# ======================================================================================

def send_frame(sock: socket.socket, obj: Dict[str, Any]) -> None:
    b = jcanon(obj)
    sock.sendall(struct.pack("!I", len(b)))
    sock.sendall(b)

def recv_frame(sock: socket.socket) -> Optional[Dict[str, Any]]:
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    (n,) = struct.unpack("!I", hdr)
    if n <= 0 or n > 16_000_000:
        return None
    data = b""
    while len(data) < n:
        chunk = sock.recv(min(65536, n - len(data)))
        if not chunk:
            return None
        data += chunk
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None

# ======================================================================================
# Node configuration and genesis
# ======================================================================================

@dataclass
class Genesis:
    version: int
    chain_id: str
    created_ms: int
    shards: int
    ma: Dict[str, Any]
    validators: List[Dict[str, Any]]  # {name, pubkey_b64, bond}
    protocol: Dict[str, Any]          # locked params
    genesis_hash: str

def make_genesis(chain_id: str, shards: int, ma_params: MAParams, validators: List[Tuple[str, str, int]], protocol: Dict[str, Any]) -> Genesis:
    created_ms = 1700000000000  # fixed time for determinism (not "now")
    g0 = {
        "version": GENESIS_VERSION,
        "chain_id": chain_id,
        "created_ms": created_ms,
        "shards": shards,
        "ma": dataclasses.asdict(ma_params),
        "validators": [{"name": n, "pubkey_b64": pk, "bond": int(b)} for (n, pk, b) in validators],
        "protocol": protocol,
    }
    gh = sha256(jcanon(g0)).hex()
    g0["genesis_hash"] = gh
    return Genesis(**g0)  # type: ignore[arg-type]

def load_genesis(path: str) -> Genesis:
    obj = json.loads(open(path, "rb").read().decode("utf-8"))
    # validate minimal keys
    for k in ("version", "chain_id", "created_ms", "shards", "ma", "validators", "protocol", "genesis_hash"):
        if k not in obj:
            raise ValueError(f"genesis missing {k}")
    # verify hash
    chk = dict(obj)
    gh = chk.pop("genesis_hash")
    if sha256(jcanon(chk)).hex() != gh:
        raise ValueError("genesis hash mismatch")
    return Genesis(**obj)

# ======================================================================================
# Minimal BFT skeleton (leader propose + votes + QC + commit)
# ======================================================================================

@dataclass
class BlockHeader:
    height: int
    round: int
    prev_hash: str
    proposer: str  # pubkey_b64
    ts_ms: int
    shard_root: str
    tx_root: str
    qc_hash: str

@dataclass
class Block:
    header: BlockHeader
    txs: List[Dict[str, Any]]

@dataclass
class Receipt:
    txid: str
    ok: bool
    reason: str
    block_height: int
    shard: int

def block_hash(b: Block) -> str:
    return sha256(jcanon({"header": dataclasses.asdict(b.header), "txs": b.txs})).hex()

def merkle_root_hex(leaves_hex: List[str]) -> str:
    if not leaves_hex:
        return sha256(b"").hex()
    level = [bytes.fromhex(x) for x in leaves_hex]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(sha256(a + b))
        level = nxt
    return level[0].hex()

# ======================================================================================
# Gossip envelopes and reliability
# ======================================================================================

CRITICAL_TYPES = {
    "PROPOSAL",
    "VOTE",
    "QC",
    "COMMIT",
    "RANDAO_COMMIT",
    "RANDAO_REVEAL",
}

@dataclass
class PendingAck:
    ack_id: str
    peer: Tuple[str, int]
    env: Dict[str, Any]
    deadline_ms: int
    retries_left: int

# ======================================================================================
# Node
# ======================================================================================

class NumchainNode:
    def __init__(
        self,
        genesis: Genesis,
        node_index: int,
        rpc_host: str,
        rpc_port: int,
        gossip_host: str,
        gossip_port: int,
        peers: List[Tuple[str, int]],
        keypair: Ed25519,
        log_path: str,
        decode_cache_max: int = 4096,
        mempool_cap: int = 250_000,
        per_sender_cap: int = 20_000,
        tx_gossip_hops: int = 4,
        gossip_timeout_ms: int = 200,
        gossip_retries: int = 2,
        max_pending_acks: int = 50_000,
        round_time_ms: int = 1000,
        block_tx_max: int = 8192,
    ):
        assert np is not None
        self.genesis = genesis
        self.node_index = node_index
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port
        self.gossip_host = gossip_host
        self.gossip_port = gossip_port
        self.peers = peers
        self.kp = keypair
        self.pub_b64 = keypair.pubkey_b64()
        self.log_path = log_path

        # locked protocol params (from genesis)
        self.shards = int(genesis.shards)
        self.ma_params = MAParams(**genesis.ma)
        self.mempool_cap = int(mempool_cap)
        self.per_sender_cap = int(per_sender_cap)
        self.block_tx_max = int(block_tx_max)
        self.round_time_ms = int(round_time_ms)

        # gossip params
        self.tx_gossip_hops = int(tx_gossip_hops)
        self.gossip_timeout_ms = int(gossip_timeout_ms)
        self.gossip_retries = int(gossip_retries)
        self.max_pending_acks = int(max_pending_acks)

        # state
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # shard-local MA states
        self.ma_state: List["np.ndarray"] = [ma_init(self.ma_params, seed=1234 + s) for s in range(self.shards)]

        # chain
        self.height = 0
        self.round = 0
        self.tip_hash = genesis.genesis_hash
        self.blocks: Dict[int, Block] = {}
        self.receipts: Dict[str, Receipt] = {}

        # mempool
        self.mempool: Dict[str, Dict[str, Any]] = {}
        self.sender_counts: Dict[str, int] = {}
        self.seen_tx = LRUSet(cap=500_000)

        # dedup envelope ids
        self.seen_env = LRUSet(cap=500_000)

        # ack/retry
        self.pending_acks: Dict[str, PendingAck] = {}
        self.ack_lock = threading.Lock()

        # counters
        self.bad_sig = 0
        self.bad_tx = 0
        self.bad_env = 0
        self.accepted_tx = 0
        self.rejected_tx = 0
        self.rej_reasons: Dict[str, int] = {}
        self.qc_success = 0
        self.qc_fail = 0
        self.commit_count = 0

        # validators
        self.validators = {v["pubkey_b64"]: v for v in genesis.validators}
        self.total_bond = sum(int(v["bond"]) for v in genesis.validators)
        if self.pub_b64 not in self.validators:
            # allow "observer" nodes
            self.validators[self.pub_b64] = {"name": f"node{node_index}", "pubkey_b64": self.pub_b64, "bond": 1}
            self.total_bond += 1

        # vote tracking for current round
        self._votes: Dict[int, Dict[str, str]] = {}  # round -> pubkey_b64 -> vote_hash
        self._proposals: Dict[int, Block] = {}

        # start network server
        self._tcp_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)

        # ack retry thread
        self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)

        # producer thread (consensus)
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)

        # replay log
        self._replay_log()

    def _log(self, rec: Dict[str, Any]) -> None:
        rec2 = dict(rec)
        rec2["ts_ms"] = now_ms()
        with open(self.log_path, "ab") as f:
            f.write(jcanon(rec2) + b"\n")

    def _replay_log(self) -> None:
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line.decode("utf-8"))
                    t = obj.get("type")
                    if t == "COMMIT":
                        b = obj["block"]
                        blk = Block(
                            header=BlockHeader(**b["header"]),
                            txs=b["txs"],
                        )
                        self._apply_committed_block(blk, replaying=True)
        except Exception:
            # If log is corrupted, user can delete it; this is a prototype.
            pass

    def start(self) -> None:
        self._tcp_thread.start()
        self._retry_thread.start()
        self._producer_thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ----------------------------
    # TX validation & submit
    # ----------------------------

    def _rej(self, reason: str) -> None:
        self.rejected_tx += 1
        self.rej_reasons[reason] = self.rej_reasons.get(reason, 0) + 1

    def submit_tx(self, tx: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Returns: (ok, reason, txid)
        tx must include:
          sender_pub_b64, sender_sig_b64, shard, layer, slot, op, ...
        """
        gh = self.genesis.genesis_hash
        try:
            sender = tx["sender_pub_b64"]
            sig_b64 = tx["sender_sig_b64"]
            shard = int(tx["shard"])
            layer = int(tx["layer"])
            slot = int(tx["slot"])
            op = tx["op"]
        except Exception:
            self.bad_tx += 1
            self._rej("bad_format")
            return False, "bad_format", ""

        if shard < 0 or shard >= self.shards:
            self.bad_tx += 1
            self._rej("bad_shard")
            return False, "bad_shard", ""
        if layer < 0 or layer >= self.ma_params.L:
            self.bad_tx += 1
            self._rej("bad_layer")
            return False, "bad_layer", ""
        if slot < 0 or slot >= self.ma_params.S:
            self.bad_tx += 1
            self._rej("bad_slot")
            return False, "bad_slot", ""
        if op not in ("add", "decay", "mix", "clip"):
            self.bad_tx += 1
            self._rej("bad_op")
            return False, "bad_op", ""

        # signature verify
        msg = tx_canonical_message(gh, {k: tx[k] for k in tx.keys() if k != "sender_sig_b64"})
        try:
            sig = b64d(sig_b64)
            pub = b64d(sender)
        except Exception:
            self.bad_sig += 1
            self._rej("bad_sig_encoding")
            return False, "bad_sig_encoding", ""

        if not Ed25519.verify(pub, msg, sig):
            self.bad_sig += 1
            self._rej("bad_sig")
            return False, "bad_sig", ""

        tid = tx_id(gh, {k: tx[k] for k in tx.keys() if k != "sender_sig_b64"})
        if self.seen_tx.seen(tid):
            self._rej("dup")
            return False, "dup", tid

        with self._lock:
            if len(self.mempool) >= self.mempool_cap:
                self._rej("mempool_full")
                return False, "mempool_full", tid
            c = self.sender_counts.get(sender, 0)
            if c >= self.per_sender_cap:
                self._rej("sender_cap")
                return False, "sender_cap", tid

            # accept
            self.mempool[tid] = tx
            self.sender_counts[sender] = c + 1
            self.seen_tx.add(tid)
            self.accepted_tx += 1
            self._log({"type": "TX_ACCEPT", "txid": tid})
            return True, "ok", tid

    # ----------------------------
    # Gossip: envelope create/verify
    # ----------------------------

    def make_env(self, mtype: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        env = {
            "v": 1,
            "chain_id": self.genesis.chain_id,
            "genesis_hash": self.genesis.genesis_hash,
            "type": mtype,
            "from": self.pub_b64,
            "ts_ms": now_ms(),
            "payload": payload,
        }
        sig = self.kp.sign(jcanon(env))
        env["sig_b64"] = b64e(sig)
        env["env_id"] = env_id(env)
        return env

    def verify_env(self, env: Dict[str, Any]) -> bool:
        try:
            if env.get("chain_id") != self.genesis.chain_id:
                return False
            if env.get("genesis_hash") != self.genesis.genesis_hash:
                return False
            sig = b64d(env["sig_b64"])
            pub = b64d(env["from"])
            # verify signature against canonical envelope without env_id
            env2 = dict(env)
            env2.pop("env_id", None)
            ok = Ed25519.verify(pub, jcanon(env2), sig)
            return ok
        except Exception:
            return False

    # ----------------------------
    # Gossip send: critical ACK + retry
    # ----------------------------

    def gossip_send(self, peer: Tuple[str, int], env: Dict[str, Any]) -> None:
        """
        Best-effort send. For critical env types, record pending ack and retry.
        """
        mtype = env["type"]
        is_critical = mtype in CRITICAL_TYPES
        # backpressure: do not queue infinite critical acks; drop non-critical first
        if is_critical:
            with self.ack_lock:
                if len(self.pending_acks) >= self.max_pending_acks:
                    # hard drop (fail closed), but count as qc_fail symptom
                    return

        try:
            s = socket.create_connection(peer, timeout=self.gossip_timeout_ms / 1000.0)
            s.settimeout(self.gossip_timeout_ms / 1000.0)
            send_frame(s, env)
            s.close()
        except Exception:
            # sender-side retry loop will handle if critical
            pass

        if is_critical:
            aid = env.get("env_id") or env_id(env)
            with self.ack_lock:
                self.pending_acks[aid] = PendingAck(
                    ack_id=aid,
                    peer=peer,
                    env=env,
                    deadline_ms=now_ms() + self.gossip_timeout_ms,
                    retries_left=self.gossip_retries,
                )

    def gossip_broadcast(self, env: Dict[str, Any], exclude: Optional[Tuple[str, int]] = None) -> None:
        for p in self.peers:
            if exclude is not None and p == exclude:
                continue
            self.gossip_send(p, env)

    def _retry_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.02)
            now = now_ms()
            todo: List[PendingAck] = []
            with self.ack_lock:
                for aid, pa in list(self.pending_acks.items()):
                    if now >= pa.deadline_ms:
                        todo.append(pa)
            for pa in todo:
                if pa.retries_left <= 0:
                    with self.ack_lock:
                        self.pending_acks.pop(pa.ack_id, None)
                    continue
                # retry send
                try:
                    s = socket.create_connection(pa.peer, timeout=self.gossip_timeout_ms / 1000.0)
                    s.settimeout(self.gossip_timeout_ms / 1000.0)
                    send_frame(s, pa.env)
                    s.close()
                except Exception:
                    pass
                with self.ack_lock:
                    cur = self.pending_acks.get(pa.ack_id)
                    if cur is None:
                        continue
                    cur.retries_left -= 1
                    cur.deadline_ms = now_ms() + self.gossip_timeout_ms

    # ----------------------------
    # TCP server
    # ----------------------------

    def _tcp_server_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.gossip_host, self.gossip_port))
        srv.listen(128)
        srv.settimeout(0.2)

        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
                t = threading.Thread(target=self._handle_conn, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception:
                continue

        try:
            srv.close()
        except Exception:
            pass

    def _handle_conn(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        try:
            conn.settimeout(self.gossip_timeout_ms / 1000.0)
            env = recv_frame(conn)
            if env is None:
                conn.close()
                return
            # ACK handling (fast path): allow unsigned ACK? keep it signed to prevent spoof/DoS confusion.
            if not isinstance(env, dict) or "type" not in env:
                conn.close()
                return

            # verify signature & dedup
            eid = env.get("env_id") or env_id(env)
            if self.seen_env.seen(eid):
                # still ACK critical to stop retries
                if env.get("type") in CRITICAL_TYPES:
                    ack = self.make_env("ACK", {"ack_id": eid})
                    send_frame(conn, ack)
                conn.close()
                return

            if not self.verify_env(env):
                self.bad_env += 1
                conn.close()
                return

            self.seen_env.add(eid)

            # For critical types: immediately ACK back on same connection
            if env["type"] in CRITICAL_TYPES:
                ack = self.make_env("ACK", {"ack_id": eid})
                send_frame(conn, ack)

            # Also accept incoming ACK to clear pending_acks
            if env["type"] == "ACK":
                aid = env["payload"].get("ack_id", "")
                if isinstance(aid, str) and aid:
                    with self.ack_lock:
                        self.pending_acks.pop(aid, None)
                conn.close()
                return

            # Handle message
            self._handle_env(env, addr)
            conn.close()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    # ----------------------------
    # Message handlers
    # ----------------------------

    def _handle_env(self, env: Dict[str, Any], addr: Tuple[str, int]) -> None:
        t = env["type"]
        pl = env.get("payload", {})

        if t == "TXGOSSIP":
            self._on_txgossip(pl, src_peer=addr)
        elif t == "PROPOSAL":
            self._on_proposal(pl, src_peer=addr)
        elif t == "VOTE":
            self._on_vote(pl, src_peer=addr)
        elif t == "QC":
            self._on_qc(pl, src_peer=addr)
        elif t == "COMMIT":
            self._on_commit(pl, src_peer=addr)
        else:
            # ignore unknown
            return

    # ----------------------------
    # TX gossip (NEW)
    # ----------------------------

    def _on_txgossip(self, pl: Dict[str, Any], src_peer: Tuple[str, int]) -> None:
        """
        Payload:
          tx: dict
          hop: int
        """
        tx = pl.get("tx")
        hop = int(pl.get("hop", 0))
        if not isinstance(tx, dict):
            return
        # Validate/accept (dedup + mempool caps)
        ok, reason, tid = self.submit_tx(tx)
        # Re-gossip only if accepted (or dup) and within hop budget.
        # If dup, still propagate if hop budget remains? Usually no; reduce chatter.
        if hop >= self.tx_gossip_hops:
            return
        if ok:
            env = self.make_env("TXGOSSIP", {"tx": tx, "hop": hop + 1})
            self.gossip_broadcast(env, exclude=src_peer)

    # ----------------------------
    # Consensus (proposal/vote/qc/commit)
    # ----------------------------

    def _choose_leader_pub(self, r: int) -> str:
        # deterministic weighted choice based on genesis hash + round
        seed = sha256((self.genesis.genesis_hash + f":leader:{r}").encode("utf-8"))
        x = int.from_bytes(seed[:8], "big")
        total = 0
        for pk, v in sorted(self.validators.items(), key=lambda kv: kv[0]):
            total += int(v.get("bond", 1))
        if total <= 0:
            return self.pub_b64
        pick = x % total
        acc = 0
        for pk, v in sorted(self.validators.items(), key=lambda kv: kv[0]):
            acc += int(v.get("bond", 1))
            if pick < acc:
                return pk
        return self.pub_b64

    def _producer_loop(self) -> None:
        # Each node runs loop; only leader proposes.
        next_tick = now_ms()
        while not self._stop.is_set():
            now = now_ms()
            if now < next_tick:
                time.sleep(0.001)
                continue
            next_tick += self.round_time_ms

            with self._lock:
                self.round += 1
                r = self.round
                leader = self._choose_leader_pub(r)

            if leader == self.pub_b64:
                self._propose_round(r)

    def _propose_round(self, r: int) -> None:
        # build block from mempool
        with self._lock:
            txids = list(self.mempool.keys())[: self.block_tx_max]
            txs = [self.mempool[tid] for tid in txids]
            prev = self.tip_hash
            height = self.height + 1

        tx_roots = [sha256(tx_canonical_message(self.genesis.genesis_hash, {k: tx[k] for k in tx if k != "sender_sig_b64"})).hex() for tx in txs]
        tx_root = merkle_root_hex(tx_roots)

        # shard_root: commit to MA states (cheap) - hash concatenated per-shard roots
        shard_roots = []
        with self._lock:
            for s in range(self.shards):
                shard_roots.append(sha256(self.ma_state[s].tobytes()).hex())
        shard_root = merkle_root_hex(shard_roots)

        hdr = BlockHeader(
            height=height,
            round=r,
            prev_hash=prev,
            proposer=self.pub_b64,
            ts_ms=now_ms(),
            shard_root=shard_root,
            tx_root=tx_root,
            qc_hash="",
        )
        blk = Block(header=hdr, txs=txs)
        bh = block_hash(blk)

        # record proposal locally
        with self._lock:
            self._proposals[r] = blk
            self._votes.setdefault(r, {})[self.pub_b64] = bh

        # gossip proposal (CRITICAL)
        env = self.make_env("PROPOSAL", {"round": r, "block": {"header": dataclasses.asdict(hdr), "txs": txs}, "block_hash": bh})
        self.gossip_broadcast(env)

        # self-vote to others as well
        venv = self.make_env("VOTE", {"round": r, "block_hash": bh, "voter": self.pub_b64})
        self.gossip_broadcast(venv)

        # try form QC after a short delay (leader side)
        threading.Thread(target=self._leader_try_qc, args=(r, bh), daemon=True).start()

    def _on_proposal(self, pl: Dict[str, Any], src_peer: Tuple[str, int]) -> None:
        r = int(pl.get("round", 0))
        b = pl.get("block")
        bh = pl.get("block_hash", "")
        if not isinstance(b, dict) or not isinstance(bh, str) or not bh:
            return
        try:
            blk = Block(header=BlockHeader(**b["header"]), txs=b["txs"])
        except Exception:
            return

        # accept proposal if leader matches
        leader = self._choose_leader_pub(r)
        if blk.header.proposer != leader:
            return
        # store proposal
        with self._lock:
            self._proposals[r] = blk

        # vote (CRITICAL)
        venv = self.make_env("VOTE", {"round": r, "block_hash": bh, "voter": self.pub_b64})
        self.gossip_broadcast(venv, exclude=src_peer)

    def _on_vote(self, pl: Dict[str, Any], src_peer: Tuple[str, int]) -> None:
        r = int(pl.get("round", 0))
        bh = pl.get("block_hash", "")
        voter = pl.get("voter", "")
        if not isinstance(bh, str) or not bh or not isinstance(voter, str):
            return
        # record vote
        with self._lock:
            self._votes.setdefault(r, {})[voter] = bh

        # if I'm leader, try QC
        if self._choose_leader_pub(r) == self.pub_b64:
            threading.Thread(target=self._leader_try_qc, args=(r, bh), daemon=True).start()

    def _leader_try_qc(self, r: int, bh: str) -> None:
        # leader collects votes for same block hash >= 2/3 bond
        with self._lock:
            votes = dict(self._votes.get(r, {}))
        total = 0
        for pk, v in self.validators.items():
            total += int(v.get("bond", 1))
        if total <= 0:
            total = 1
        have = 0
        signers = []
        for voter, vb in votes.items():
            if vb != bh:
                continue
            bond = int(self.validators.get(voter, {}).get("bond", 1))
            have += bond
            signers.append(voter)

        if have * 3 < total * 2:
            return

        qc = {"round": r, "block_hash": bh, "signers": sorted(signers), "bond_yes": have, "bond_total": total}
        qc_hash = sha256(jcanon(qc)).hex()

        env = self.make_env("QC", {"qc": qc, "qc_hash": qc_hash})
        self.gossip_broadcast(env)

        # commit immediately (HotStuff-ish single-step)
        cenv = self.make_env("COMMIT", {"round": r, "qc_hash": qc_hash, "block_hash": bh})
        self.gossip_broadcast(cenv)

        self.qc_success += 1

    def _on_qc(self, pl: Dict[str, Any], src_peer: Tuple[str, int]) -> None:
        # currently informational; commit handler does state update
        return

    def _on_commit(self, pl: Dict[str, Any], src_peer: Tuple[str, int]) -> None:
        r = int(pl.get("round", 0))
        bh = pl.get("block_hash", "")
        if not isinstance(bh, str) or not bh:
            return
        with self._lock:
            blk = self._proposals.get(r)
        if blk is None:
            return
        if block_hash(blk) != bh:
            return
        # apply commit
        self._apply_committed_block(blk, replaying=False)
        self.commit_count += 1
        self._log({"type": "COMMIT", "block": {"header": dataclasses.asdict(blk.header), "txs": blk.txs}})

    def _apply_committed_block(self, blk: Block, replaying: bool) -> None:
        # apply txs to MA per shard; build receipts; remove from mempool
        gh = self.genesis.genesis_hash
        applied = 0
        with self._lock:
            self.height = max(self.height, blk.header.height)
            self.tip_hash = block_hash(blk)

            for tx in blk.txs:
                tid = tx_id(gh, {k: tx[k] for k in tx if k != "sender_sig_b64"})
                shard = int(tx["shard"])
                try:
                    ma_apply_tx(self.ma_state[shard], self.ma_params, tx)
                    ok = True
                    reason = "ok"
                except Exception:
                    ok = False
                    reason = "apply_fail"
                self.receipts[tid] = Receipt(txid=tid, ok=ok, reason=reason, block_height=blk.header.height, shard=shard)

                # clear mempool entry if present
                if tid in self.mempool:
                    sender = self.mempool[tid].get("sender_pub_b64", "")
                    self.mempool.pop(tid, None)
                    if sender:
                        self.sender_counts[sender] = max(0, self.sender_counts.get(sender, 1) - 1)
                applied += 1

            self.blocks[blk.header.height] = blk

    # ==================================================================================
    # RPC interface helpers
    # ==================================================================================

    def get_health(self) -> Dict[str, Any]:
        rss = 0.0
        if psutil is not None:
            try:
                rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            except Exception:
                rss = 0.0
        with self._lock:
            return {
                "package_version": VERSION,
                "chain_id": self.genesis.chain_id,
                "genesis_hash": self.genesis.genesis_hash,
                "node_pub": self.pub_b64,
                "height": self.height,
                "round": self.round,
                "tip": self.tip_hash[:12],
                "mempool": len(self.mempool),
                "accepted_tx": self.accepted_tx,
                "rejected_tx": self.rejected_tx,
                "bad_sig": self.bad_sig,
                "bad_env": self.bad_env,
                "qc_success": self.qc_success,
                "commit_count": self.commit_count,
                "rej_reasons": dict(self.rej_reasons),
                "pending_acks": len(self.pending_acks),
                "rss_mb": rss,
            }

    def get_metrics_text(self) -> str:
        h = self.get_health()
        lines = []
        def m(name: str, val: Any, labels: Optional[Dict[str, str]] = None) -> None:
            if labels:
                lab = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                lines.append(f"{name}{{{lab}}} {val}")
            else:
                lines.append(f"{name} {val}")

        labels = {"chain_id": self.genesis.chain_id, "node": self.pub_b64[:12], "version": VERSION}
        lines.append("# HELP numchain_info build info")
        lines.append("# TYPE numchain_info gauge")
        m("numchain_info", 1, labels)
        lines.append("# HELP numchain_height chain height")
        lines.append("# TYPE numchain_height gauge")
        m("numchain_height", h["height"], labels)
        lines.append("# HELP numchain_mempool txs in mempool")
        lines.append("# TYPE numchain_mempool gauge")
        m("numchain_mempool", h["mempool"], labels)
        lines.append("# HELP numchain_accepted_tx accepted tx total")
        lines.append("# TYPE numchain_accepted_tx counter")
        m("numchain_accepted_tx", h["accepted_tx"], labels)
        lines.append("# HELP numchain_rejected_tx rejected tx total")
        lines.append("# TYPE numchain_rejected_tx counter")
        m("numchain_rejected_tx", h["rejected_tx"], labels)
        lines.append("# HELP numchain_bad_sig bad signatures total")
        lines.append("# TYPE numchain_bad_sig counter")
        m("numchain_bad_sig", h["bad_sig"], labels)
        lines.append("# HELP numchain_qc_success qc successes")
        lines.append("# TYPE numchain_qc_success counter")
        m("numchain_qc_success", h["qc_success"], labels)
        lines.append("# HELP numchain_commit_count commits applied")
        lines.append("# TYPE numchain_commit_count counter")
        m("numchain_commit_count", h["commit_count"], labels)
        lines.append("# HELP numchain_pending_acks pending critical message ACKs")
        lines.append("# TYPE numchain_pending_acks gauge")
        m("numchain_pending_acks", h["pending_acks"], labels)
        lines.append("# HELP numchain_rss_mb resident set size")
        lines.append("# TYPE numchain_rss_mb gauge")
        m("numchain_rss_mb", h["rss_mb"], labels)

        # Rejection reasons as labeled counters
        lines.append("# HELP numchain_reject_reason rejected tx reasons")
        lines.append("# TYPE numchain_reject_reason counter")
        for k, v in h["rej_reasons"].items():
            m("numchain_reject_reason", v, {**labels, "reason": str(k)})

        return "\n".join(lines) + "\n"

# ======================================================================================
# RPC server (FastAPI)
# ======================================================================================

def make_app(node: NumchainNode) -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("FastAPI not available. Run: python numchain_v142.py deps")
    app = FastAPI(title=f"NUMCHAIN {VERSION}", version=VERSION)

    @app.get("/health")
    def health():
        return node.get_health()

    @app.get("/metrics", response_class=PlainTextResponse)  # type: ignore[misc]
    def metrics():
        return node.get_metrics_text()

    @app.post("/tx")
    def post_tx(body: Dict[str, Any]):
        if not isinstance(body, dict) or "tx" not in body:
            raise HTTPException(status_code=400, detail="body must be {tx:{...}}")
        tx = body["tx"]
        if not isinstance(tx, dict):
            raise HTTPException(status_code=400, detail="tx must be object")
        ok, reason, tid = node.submit_tx(tx)
        if ok:
            # NEW: broadcast tx gossip from RPC accept point
            env = node.make_env("TXGOSSIP", {"tx": tx, "hop": 0})
            node.gossip_broadcast(env)
        return {"ok": ok, "reason": reason, "txid": tid}

    @app.get("/block/{height}")
    def get_block(height: int):
        with node._lock:
            blk = node.blocks.get(int(height))
        if blk is None:
            raise HTTPException(status_code=404, detail="not found")
        return {"header": dataclasses.asdict(blk.header), "txs": blk.txs}

    @app.get("/receipt/{txid}")
    def get_receipt(txid: str):
        with node._lock:
            rc = node.receipts.get(txid)
        if rc is None:
            raise HTTPException(status_code=404, detail="not found")
        return dataclasses.asdict(rc)

    return app

# ======================================================================================
# Tools: keygen, genesis, flood, poll
# ======================================================================================

def ensure_deps() -> None:
    import subprocess
    pkgs = [
        "numpy",
        "psutil",
        "fastapi",
        "uvicorn[standard]",
        "httpx",
        "pynacl",
        "cryptography",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.check_call(cmd)

def keygen(n: int) -> List[Tuple[str, str]]:
    out = []
    for i in range(n):
        kp = Ed25519()
        out.append((kp.privkey_b64, kp.pubkey_b64()))
    return out

def build_synthetic_tx(genesis_hash: str, sender_kp: Ed25519, shard: int, L: int, S: int, D: int, q: int) -> Dict[str, Any]:
    # create an "add" tx with small int16 vector
    rng = random.Random(int.from_bytes(sha256((sender_kp.pubkey_b64() + str(now_ms())).encode()).digest()[:8], "big"))
    layer = rng.randrange(0, L)
    slot = rng.randrange(0, S)

    # small sparse-ish delta
    v = [0] * D
    for _ in range(8):
        j = rng.randrange(0, D)
        v[j] = rng.randrange(-32, 33)
    vb = struct.pack(f"<{D}h", *v)
    vec_b64 = b64e(vb)

    tx_wo_sig = {
        "sender_pub_b64": sender_kp.pubkey_b64(),
        "shard": int(shard),
        "layer": int(layer),
        "slot": int(slot),
        "op": "add",
        "vec_b64": vec_b64,
        "delta_norm_max": int(q * 4),
        "nonce": rng.getrandbits(64),
    }
    msg = tx_canonical_message(genesis_hash, tx_wo_sig)
    sig = sender_kp.sign(msg)
    tx = dict(tx_wo_sig)
    tx["sender_sig_b64"] = b64e(sig)
    return tx

def flood_rpc(url: str, genesis_path: str, seconds: int, target_tps: int, senders: int, ramp_up: int) -> None:
    if httpx is None:
        raise RuntimeError("httpx missing. Run deps.")
    g = load_genesis(genesis_path)
    ma = MAParams(**g.ma)
    sender_keys = [Ed25519() for _ in range(senders)]
    t_end = time.time() + seconds
    sent = 0
    okc = 0
    rej = 0
    lat = []
    reasons: Dict[str, int] = {}

    client = httpx.Client(timeout=5.0)

    start = time.time()
    def current_tps() -> float:
        if ramp_up <= 0:
            return float(target_tps)
        elapsed = time.time() - start
        if elapsed >= ramp_up:
            return float(target_tps)
        return float(target_tps) * (elapsed / float(ramp_up))

    next_tick = time.time()
    interval = 1.0 / max(1, target_tps)

    while time.time() < t_end:
        tps = current_tps()
        interval = 1.0 / max(1.0, tps)
        if time.time() < next_tick:
            time.sleep(min(0.001, next_tick - time.time()))
            continue
        next_tick += interval

        sk = sender_keys[sent % senders]
        shard = sent % max(1, g.shards)
        tx = build_synthetic_tx(g.genesis_hash, sk, shard, ma.L, ma.S, ma.D, ma.q)
        t0 = time.time()
        try:
            r = client.post(url.rstrip("/") + "/tx", json={"tx": tx})
            dt = (time.time() - t0) * 1000.0
            lat.append(dt)
            j = r.json()
            if j.get("ok"):
                okc += 1
            else:
                rej += 1
                rsn = str(j.get("reason", "unknown"))
                reasons[rsn] = reasons.get(rsn, 0) + 1
        except Exception:
            rej += 1
            reasons["rpc_error"] = reasons.get("rpc_error", 0) + 1
        sent += 1

    client.close()

    dur = seconds
    lat.sort()
    def pct(p: float) -> float:
        if not lat:
            return 0.0
        idx = int((p / 100.0) * (len(lat) - 1))
        return lat[idx]

    print("\n" + "=" * 94)
    print("NUMCHAIN · RPC Flood Summary")
    print("=" * 94)
    print(f"url={url} seconds={seconds} target_tps={target_tps} senders={senders} ramp_up={ramp_up}")
    print(f"sent={sent} ok={okc} rej={rej} accepted_tps={okc/dur:.1f}")
    print(f"lat_ms p50={pct(50):.3f} p95={pct(95):.3f} p99={pct(99):.3f} max={lat[-1]:.3f}" if lat else "lat_ms n/a")
    if reasons:
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:12]
        print("rejection_reasons(top): " + ", ".join([f"{k}={v}" for k, v in top]))
    print("=" * 94 + "\n")

def poll_nodes(urls: List[str], seconds: int, every: float) -> None:
    if httpx is None:
        raise RuntimeError("httpx missing. Run deps.")
    client = httpx.Client(timeout=2.0)
    t_end = time.time() + seconds
    while time.time() < t_end:
        hs = []
        tips = []
        mem = []
        pend = []
        for u in urls:
            try:
                r = client.get(u.rstrip("/") + "/health")
                j = r.json()
                hs.append(int(j.get("height", 0)))
                tips.append(str(j.get("tip", "")))
                mem.append(int(j.get("mempool", 0)))
                pend.append(int(j.get("pending_acks", 0)))
            except Exception:
                hs.append(-1)
                tips.append("ERR")
                mem.append(-1)
                pend.append(-1)
        good = [h for h in hs if h >= 0]
        spread = (max(good) - min(good)) if good else -1
        print(f"[poll] heights={hs} spread={spread} mem={mem} pending_acks={pend} tips={[t[:6] for t in tips]}")
        time.sleep(every)
    client.close()

# ======================================================================================
# Main CLI
# ======================================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="numchain_v142", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("deps", help="pip install runtime deps")

    pk = sub.add_parser("keygen", help="generate ed25519 keys")
    pk.add_argument("--n", type=int, default=5)

    pg = sub.add_parser("genesis", help="write deterministic genesis")
    pg.add_argument("--out", type=str, required=True)
    pg.add_argument("--nodes", type=int, default=5)
    pg.add_argument("--chain-id", type=str, default=CHAIN_ID_DEFAULT)
    pg.add_argument("--shards", type=int, default=8)
    pg.add_argument("--bond", type=int, default=1000)

    pr = sub.add_parser("run", help="run a node (gossip + rpc)")
    pr.add_argument("--genesis", type=str, required=True)
    pr.add_argument("--node-index", type=int, default=0)
    pr.add_argument("--rpc", type=int, default=8000)
    pr.add_argument("--rpc-host", type=str, default="0.0.0.0")
    pr.add_argument("--gossip", type=int, default=9000)
    pr.add_argument("--gossip-host", type=str, default="0.0.0.0")
    pr.add_argument("--peers", type=str, default="", help="comma list host:port")
    pr.add_argument("--log", type=str, default="", help="append-only jsonl log path (default ./nodeX.log)")
    pr.add_argument("--privkey-b64", type=str, default="", help="optional fixed privkey for node identity")
    pr.add_argument("--mempool-cap", type=int, default=250_000)
    pr.add_argument("--per-sender-cap", type=int, default=20_000)
    pr.add_argument("--round-time-ms", type=int, default=1000)
    pr.add_argument("--block-tx-max", type=int, default=8192)
    pr.add_argument("--tx-gossip-hops", type=int, default=4)
    pr.add_argument("--gossip-timeout-ms", type=int, default=200)
    pr.add_argument("--gossip-retries", type=int, default=2)
    pr.add_argument("--max-pending-acks", type=int, default=50_000)

    pf = sub.add_parser("flood", help="rpc flood a node")
    pf.add_argument("--url", type=str, required=True)
    pf.add_argument("--genesis", type=str, required=True)
    pf.add_argument("--seconds", type=int, default=60)
    pf.add_argument("--target-tps", type=int, default=4000)
    pf.add_argument("--senders", type=int, default=256)
    pf.add_argument("--ramp-up", type=int, default=0)

    pp = sub.add_parser("poll", help="poll /health across nodes")
    pp.add_argument("--urls", type=str, required=True)
    pp.add_argument("--seconds", type=int, default=60)
    pp.add_argument("--every", type=float, default=2.0)

    return ap

def main() -> None:
    if np is None:
        # Don't hard fail on commands that don't need numpy.
        pass

    ap = build_parser()
    # Jupyter/Colab safe parse: ignore unknown "-f kernel.json" etc.
    args, _unknown = ap.parse_known_args()

    if args.cmd == "deps":
        ensure_deps()
        print("deps installed")
        return

    if args.cmd == "keygen":
        ks = keygen(args.n)
        for i, (sk, pk) in enumerate(ks):
            print(json.dumps({"i": i, "privkey_b64": sk, "pubkey_b64": pk}))
        return

    if args.cmd == "genesis":
        if np is None:
            print("ERROR: numpy required. Run: python numchain_v142.py deps", file=sys.stderr)
            sys.exit(2)
        # Deterministic validator keys for genesis: derive from chain_id + index via HMAC
        validators = []
        for i in range(args.nodes):
            seed = hmac.new(args.chain_id.encode("utf-8"), f"validator:{i}".encode("utf-8"), hashlib.sha256).digest()[:32]
            kp = Ed25519(seed)
            validators.append((f"val{i}", kp.pubkey_b64(), int(args.bond)))
        proto = {
            "locked": True,
            "mempool_cap": 250_000,
            "per_sender_cap": 20_000,
            "round_time_ms": 1000,
            "block_tx_max": 8192,
            "tx_gossip_hops": 4,
            "gossip_timeout_ms": 200,
            "gossip_retries": 2,
            "max_pending_acks": 50_000,
        }
        ma = MAParams()
        g = make_genesis(args.chain_id, int(args.shards), ma, validators, proto)
        with open(args.out, "wb") as f:
            f.write((json.dumps(dataclasses.asdict(g), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        print(f"wrote genesis {args.out}")
        print(f"genesis_hash={g.genesis_hash}")
        return

    if args.cmd == "run":
        if np is None or FastAPI is None or uvicorn is None:
            print("ERROR: missing deps. Run: python numchain_v142.py deps", file=sys.stderr)
            sys.exit(2)
        g = load_genesis(args.genesis)

        # Identity: if privkey provided, use it; else deterministic from genesis + node index
        if args.privkey_b64:
            sk = b64d(args.privkey_b64)
        else:
            sk = hmac.new(g.genesis_hash.encode("utf-8"), f"node:{args.node_index}".encode("utf-8"), hashlib.sha256).digest()[:32]
        kp = Ed25519(sk)

        peers = []
        if args.peers.strip():
            for item in args.peers.split(","):
                item = item.strip()
                if not item:
                    continue
                host, port = item.split(":")
                peers.append((host, int(port)))

        logp = args.log.strip() or f"./node{args.node_index}.log"

        node = NumchainNode(
            genesis=g,
            node_index=int(args.node_index),
            rpc_host=args.rpc_host,
            rpc_port=int(args.rpc),
            gossip_host=args.gossip_host,
            gossip_port=int(args.gossip),
            peers=peers,
            keypair=kp,
            log_path=logp,
            mempool_cap=int(args.mempool_cap),
            per_sender_cap=int(args.per_sender_cap),
            tx_gossip_hops=int(args.tx_gossip_hops),
            gossip_timeout_ms=int(args.gossip_timeout_ms),
            gossip_retries=int(args.gossip_retries),
            max_pending_acks=int(args.max_pending_acks),
            round_time_ms=int(args.round_time_ms),
            block_tx_max=int(args.block_tx_max),
        )
        node.start()

        app = make_app(node)
        print("=" * 94)
        print(f"NUMCHAIN {VERSION} node_index={args.node_index} pub={node.pub_b64[:16]} chain_id={g.chain_id}")
        print(f"RPC   http://{args.rpc_host}:{args.rpc}")
        print(f"GOSSIP {args.gossip_host}:{args.gossip} peers={len(peers)}")
        print(f"LOG   {logp}")
        print("=" * 94)
        uvicorn.run(app, host=args.rpc_host, port=int(args.rpc), log_level="info")
        return

    if args.cmd == "flood":
        flood_rpc(args.url, args.genesis, int(args.seconds), int(args.target_tps), int(args.senders), int(args.ramp_up))
        return

    if args.cmd == "poll":
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        poll_nodes(urls, int(args.seconds), float(args.every))
        return

    raise RuntimeError("unreachable")

if __name__ == "__main__":
    main()
```0
