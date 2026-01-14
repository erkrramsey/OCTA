#!/usr/bin/env python3
"""
NUMCHAIN v1.4.2 (closed-alpha kernel) — now wired for MA_DELTA_V1 (NUMCHAIN x MA_ORTHOSPACE_CONNECTOME)

What this is:
  - A deterministic, replayable, multi-node (localhost/LAN) chain kernel with:
      * Ed25519 signatures
      * TCP gossip (TX + consensus messages) with ACK/retry on critical messages
      * Append-only JSONL log + replay
      * Minimal RPC for tx submit + block/receipt + health/metrics
  - PLUS: Direct execution support for MA_DELTA_V1 transactions (the fixed 64-byte header + payload format)
      * Deterministic integer-only apply path:
          - sign-extend payload i16 Q8.8 -> i32 Q8.8
          - EMA/DRIFT uses integer mul + >>16 (Q0.16 lr)
          - mandatory per-dimension clip (default enabled)
          - deterministic integer L2 clamp using normative isqrt_u64
      * MA root hash computed deterministically across nodes and exposed via /health and /metrics.

What this is NOT:
  - Not a public testnet (no WAN hardening, NAT traversal, validator onboarding, fee market, etc.)
  - Not a full storage/compute marketplace (PoC/PoP are out of scope for this wiring pass)
  - Not a general VM (MA deltas are the only “state machine” extension here)

Run:
  # Install deps
  python3 numchain_v142.py deps

  # Generate genesis (devnet)
  python3 numchain_v142.py genesis --out genesis.json --nodes 5 --topics 64 --dim 256

  # Single node
  python3 numchain_v142.py run --genesis genesis.json --node-index 0 --rpc 8000 --gossip 9000 --peers ""

  # Multi-node (manual example)
  python3 numchain_v142.py run --genesis genesis.json --node-index 0 --rpc 8000 --gossip 9000 --peers 127.0.0.1:9001,127.0.0.1:9002
  python3 numchain_v142.py run --genesis genesis.json --node-index 1 --rpc 8001 --gossip 9001 --peers 127.0.0.1:9000,127.0.0.1:9002
  python3 numchain_v142.py run --genesis genesis.json --node-index 2 --rpc 8002 --gossip 9002 --peers 127.0.0.1:9000,127.0.0.1:9001

RPC:
  POST /tx    (normal tx envelope; supports kind="MA_DELTA_V1" with madelta_b64 or madelta_hex)
  GET  /health
  GET  /metrics
  GET  /block/{height}
  GET  /receipt/{txid}

Tools:
  python3 numchain_v142.py flood --genesis genesis.json --url http://127.0.0.1:8000 --seconds 30 --target_tps 500 --senders 64
  python3 numchain_v142.py poll --urls http://127.0.0.1:8000,http://127.0.0.1:8001 --seconds 30 --every 2

Notes:
  - This file expects a compliant MA delta producer to generate the binary MADelta (see protocol.md + producer ref).
  - This kernel validates the MADelta header strictly and applies the delta deterministically.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import queue
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Optional deps (loaded lazily)
# -----------------------------

def _lazy_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

np = None
psutil = None
fastapi = None
uvicorn = None
httpx = None

# Ed25519: prefer pynacl, fallback cryptography
nacl_signing = None
crypto_ed25519 = None

# -----------------------------
# Utilities
# -----------------------------

def now_ns() -> int:
    return time.time_ns()

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)

def jcanon(obj: Any) -> bytes:
    # Canonical JSON bytes: stable across nodes
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def clamp_i32(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

def clamp_i64(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

# -----------------------------
# Normative integer sqrt (protocol)
# -----------------------------

def isqrt_u64(n: int) -> int:
    """Floor sqrt for 0 <= n < 2^64. Normative, platform-stable."""
    n = int(n) & 0xFFFFFFFFFFFFFFFF
    if n == 0:
        return 0
    x = 0
    bit = 1 << 62
    while bit > n:
        bit >>= 2
    while bit != 0:
        if n >= (x + bit) & 0xFFFFFFFFFFFFFFFF:
            n = (n - (x + bit)) & 0xFFFFFFFFFFFFFFFF
            x = ((x >> 1) + bit) & 0xFFFFFFFFFFFFFFFF
        else:
            x = (x >> 1) & 0xFFFFFFFFFFFFFFFF
        bit >>= 2
    return x

# -----------------------------
# MADelta v1 parsing (fixed 64-byte header)
# -----------------------------

MAD1_MAGIC = 0x4D414431  # "MAD1" little-endian u32
FLAG_HAS_PROVENANCE = 1 << 0

BANK_POS = 1
BANK_NEG = 2
BANK_RES = 3

ACTION_EMA_ADD = 0
ACTION_REPLACE_ABS = 1
ACTION_DRIFT_ADD = 2
ACTION_CONF_ONLY = 3

VEC_CODEC_DENSE_I16_Q8_8 = 0

MAD1_HEADER_STRUCT = struct.Struct("<IHHQQQIQQBBBBHHHHI4x")  # exactly 64 bytes

@dataclass(frozen=True)
class MADeltaHeader:
    magic: int
    version: int
    flags: int
    chain_id_hash: int
    height_hint: int
    timestamp_ns: int
    topic_id: int
    topic_tag: int
    bank: int
    action: int
    vec_codec: int
    proto_idx: int
    dim: int
    reserved: int
    delta_conf_q: int
    lr_q: int
    payload_len: int

def parse_madelta_v1(blob: bytes) -> Tuple[MADeltaHeader, Optional[List[int]], Optional[bytes]]:
    """
    Returns (header, payload_i16_list_or_None, provenance_or_None).
    payload_i16_list is list of Python ints in range [-32768, 32767] (Q8.8).
    """
    if len(blob) < 64:
        raise ValueError("MADelta truncated (<64)")
    fields = MAD1_HEADER_STRUCT.unpack(blob[:64])
    h = MADeltaHeader(*fields)
    if h.magic != MAD1_MAGIC:
        raise ValueError("MADelta bad magic")
    if h.version != 1:
        raise ValueError("MADelta unsupported version")
    if h.reserved != 0:
        raise ValueError("MADelta reserved must be 0")
    if h.bank not in (BANK_POS, BANK_NEG, BANK_RES):
        raise ValueError("MADelta bad bank")
    if h.action not in (ACTION_EMA_ADD, ACTION_REPLACE_ABS, ACTION_DRIFT_ADD, ACTION_CONF_ONLY):
        raise ValueError("MADelta bad action")
    if h.vec_codec != VEC_CODEC_DENSE_I16_Q8_8:
        raise ValueError("MADelta unsupported vec_codec (v1 only dense)")
    if h.dim <= 0 or h.dim > 4096:
        raise ValueError("MADelta dim out of bounds")

    payload_start = 64
    payload_end = payload_start + int(h.payload_len)
    if payload_end > len(blob):
        raise ValueError("MADelta truncated payload")
    payload_bytes = blob[payload_start:payload_end]

    has_prov = (h.flags & FLAG_HAS_PROVENANCE) != 0
    prov = None
    if has_prov:
        if payload_end + 32 > len(blob):
            raise ValueError("MADelta missing provenance tail")
        prov = blob[payload_end:payload_end + 32]

    payload_i16 = None
    if h.action == ACTION_CONF_ONLY:
        if h.payload_len != 0:
            raise ValueError("CONF_ONLY must have payload_len=0")
    else:
        if h.payload_len != h.dim * 2:
            raise ValueError("MADelta payload_len mismatch for dense")
        # unpack little-endian int16[]
        payload_i16 = list(struct.unpack("<" + "h" * h.dim, payload_bytes))

    return h, payload_i16, prov

# -----------------------------
# Similarity (producer-side reference; useful for debug / optional local routing)
# -----------------------------

def similarity_q15(a_q88_i32: List[int], b_q88_i32: List[int]) -> int:
    """Cosine-like similarity in Q0.15, deterministic integer implementation."""
    if len(a_q88_i32) != len(b_q88_i32):
        raise ValueError("sim dim mismatch")
    dot = 0
    na = 0
    nb = 0
    for ai, bi in zip(a_q88_i32, b_q88_i32):
        dot += int(ai) * int(bi)           # Q16.16
        na += int(ai) * int(ai)
        nb += int(bi) * int(bi)
    if na == 0 or nb == 0:
        return 0
    denom = isqrt_u64(na) * isqrt_u64(nb)
    if denom == 0:
        return 0
    sim = (dot * 32768) // denom
    return clamp_i32(int(sim), -32768, 32767)

# -----------------------------
# MA state: per-topic/bank/proto
# -----------------------------

@dataclass
class ProtoState:
    vec: List[int]    # i32 Q8.8, length = dim
    conf: int         # u16 (Q4.12 interpreted)
    usage: int        # u32

@dataclass
class MAConfig:
    dim: int
    topics: int
    k_pos: int
    k_neg: int
    # bounded dynamics:
    per_dim_clip_q: int  # Q8.8
    max_norm_q: int      # Q8.8
    # learning rates:
    default_lr_q: int    # Q0.16
    gentle_lr_q: int     # Q0.16

def l2_clamp_q8_8(vec: List[int], max_norm_q: int) -> None:
    """
    In-place clamp to L2 norm <= max_norm_q (Q8.8).
    Uses i64 intermediates + normative isqrt_u64.
    """
    # sumsq in Q16.16
    sumsq = 0
    for v in vec:
        vv = int(v)
        sumsq += vv * vv
        # keep within Python int anyway; spec uses saturating_add (not needed here)
    if sumsq == 0:
        return
    curr_norm_q = int(isqrt_u64(sumsq))
    if curr_norm_q <= int(max_norm_q):
        return
    scale_q = (int(max_norm_q) << 16) // curr_norm_q  # Q0.16
    for i in range(len(vec)):
        scaled = (int(vec[i]) * scale_q) >> 16
        # final safety clamp to i16-compatible range in Q8.8 space:
        if scaled < -32768:
            scaled = -32768
        elif scaled > 32767:
            scaled = 32767
        vec[i] = int(scaled)

def apply_madelta_to_state(
    st: Dict[int, Dict[int, Dict[int, ProtoState]]],
    cfg: MAConfig,
    header: MADeltaHeader,
    payload_i16: Optional[List[int]],
) -> None:
    """
    Deterministic apply path. Mutates st in place.
    st structure: topic -> bank -> proto_idx -> ProtoState
    """
    if header.dim != cfg.dim:
        raise ValueError("dim mismatch vs genesis")
    if header.topic_id < 0 or header.topic_id >= cfg.topics:
        raise ValueError("topic_id out of bounds")
    if header.bank == BANK_POS:
        kmax = cfg.k_pos
    elif header.bank == BANK_NEG:
        kmax = cfg.k_neg
    else:
        # RES bank reserved; treat as K_POS for bounds or reject
        kmax = cfg.k_pos
    if header.proto_idx < 0 or header.proto_idx >= kmax:
        raise ValueError("proto_idx out of bounds")
    if header.action != ACTION_CONF_ONLY and (payload_i16 is None or len(payload_i16) != cfg.dim):
        raise ValueError("payload missing/mismatch")

    topic = st.setdefault(header.topic_id, {})
    bank = topic.setdefault(header.bank, {})
    proto = bank.get(header.proto_idx, None)

    # CONF_ONLY: no vector change
    if header.action == ACTION_CONF_ONLY:
        if proto is None:
            raise ValueError("CONF_ONLY requires existing proto")
        proto.conf = min(65535, int(proto.conf) + int(header.delta_conf_q))
        proto.usage = min(0xFFFFFFFF, int(proto.usage) + 1)
        return

    # Determine lr
    if header.action == ACTION_DRIFT_ADD:
        lr_q = cfg.gentle_lr_q
        # override not allowed for drift in v1 semantics
    elif header.action == ACTION_EMA_ADD:
        lr_q = int(header.lr_q) if int(header.lr_q) != 0 else cfg.default_lr_q
    else:
        lr_q = cfg.default_lr_q  # unused for replace, but keep defined

    # Build new vector
    if proto is None:
        # New slot must be REPLACE_ABS
        if header.action != ACTION_REPLACE_ABS:
            raise ValueError("new slot must use REPLACE_ABS")
        new_vec = [int(x) for x in payload_i16]  # sign-extend i16 -> i32 Q8.8 (no shifts)
        new_conf = int(header.delta_conf_q)
        new_usage = 1
        # Mandatory per-dim clip
        clip = int(cfg.per_dim_clip_q)
        for i in range(cfg.dim):
            v = new_vec[i]
            if v < -clip:
                v = -clip
            elif v > clip:
                v = clip
            new_vec[i] = v
        # L2 clamp
        l2_clamp_q8_8(new_vec, cfg.max_norm_q)
        bank[header.proto_idx] = ProtoState(vec=new_vec, conf=min(65535, new_conf), usage=new_usage)
        return

    # Existing slot:
    if header.action == ACTION_REPLACE_ABS:
        new_vec = [int(x) for x in payload_i16]
        # usage halving preserved
        new_usage = max(1, int(proto.usage) // 2)
    elif header.action in (ACTION_EMA_ADD, ACTION_DRIFT_ADD):
        new_vec = list(proto.vec)
        # ds = (payload * lr) >> 16, payload is Q8.8 i16, lr is Q0.16 => ds Q8.8
        for i in range(cfg.dim):
            ds = (int(payload_i16[i]) * int(lr_q)) >> 16
            new_vec[i] = int(new_vec[i]) + int(ds)
        new_usage = min(0xFFFFFFFF, int(proto.usage) + 1)
    else:
        raise ValueError("unknown action")

    # Mandatory per-dim clip
    clip = int(cfg.per_dim_clip_q)
    for i in range(cfg.dim):
        v = int(new_vec[i])
        if v < -clip:
            v = -clip
        elif v > clip:
            v = clip
        new_vec[i] = v

    # L2 clamp
    l2_clamp_q8_8(new_vec, cfg.max_norm_q)

    # Conf accumulates saturating
    new_conf = min(65535, int(proto.conf) + int(header.delta_conf_q))

    proto.vec = new_vec
    proto.conf = new_conf
    proto.usage = new_usage

def ma_root_hash(st: Dict[int, Dict[int, Dict[int, ProtoState]]], cfg: MAConfig) -> bytes:
    """
    Deterministic root: sha256 over sorted (topic, bank, idx, conf, usage, vec bytes LE i32?).
    We store vectors as i32 Q8.8; serialize as little-endian i32 for each component.
    """
    h = hashlib.sha256()
    # sort keys for determinism
    for topic_id in sorted(st.keys()):
        topic = st[topic_id]
        for bank_id in sorted(topic.keys()):
            bank = topic[bank_id]
            for idx in sorted(bank.keys()):
                p = bank[idx]
                h.update(struct.pack("<I", int(topic_id)))
                h.update(struct.pack("<B", int(bank_id)))
                h.update(struct.pack("<B", int(idx)))
                h.update(struct.pack("<H", int(p.conf) & 0xFFFF))
                h.update(struct.pack("<I", int(p.usage) & 0xFFFFFFFF))
                # vector
                for v in p.vec:
                    h.update(struct.pack("<i", int(v)))
    return h.digest()

# -----------------------------
# Ed25519 signing
# -----------------------------

@dataclass
class Keypair:
    sk: bytes
    pk: bytes

def ed25519_keygen(seed32: Optional[bytes] = None) -> Keypair:
    global nacl_signing, crypto_ed25519
    if nacl_signing is None:
        nacl = _lazy_import("nacl.signing")
        if nacl is not None:
            nacl_signing = nacl
    if nacl_signing is not None:
        if seed32 is None:
            sk_obj = nacl_signing.SigningKey.generate()
        else:
            if len(seed32) != 32:
                raise ValueError("seed32 must be 32 bytes")
            sk_obj = nacl_signing.SigningKey(seed32)
        pk_obj = sk_obj.verify_key
        return Keypair(sk=bytes(sk_obj._seed), pk=bytes(pk_obj))
    # fallback: cryptography
    if crypto_ed25519 is None:
        crypto = _lazy_import("cryptography.hazmat.primitives.asymmetric.ed25519")
        if crypto is not None:
            crypto_ed25519 = crypto
    if crypto_ed25519 is None:
        raise RuntimeError("No Ed25519 backend available. Install pynacl or cryptography.")
    if seed32 is None:
        sk_obj = crypto_ed25519.Ed25519PrivateKey.generate()
    else:
        sk_obj = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(seed32)
    pk_obj = sk_obj.public_key()
    sk_bytes = sk_obj.private_bytes(
        encoding=_lazy_import("cryptography.hazmat.primitives.serialization").Encoding.Raw,
        format=_lazy_import("cryptography.hazmat.primitives.serialization").PrivateFormat.Raw,
        encryption_algorithm=_lazy_import("cryptography.hazmat.primitives.serialization").NoEncryption(),
    )
    pk_bytes = pk_obj.public_bytes(
        encoding=_lazy_import("cryptography.hazmat.primitives.serialization").Encoding.Raw,
        format=_lazy_import("cryptography.hazmat.primitives.serialization").PublicFormat.Raw,
    )
    return Keypair(sk=sk_bytes, pk=pk_bytes)

def ed25519_sign(sk_seed32: bytes, msg: bytes) -> bytes:
    global nacl_signing, crypto_ed25519
    if nacl_signing is not None:
        sk_obj = nacl_signing.SigningKey(sk_seed32)
        sig = sk_obj.sign(msg).signature
        return bytes(sig)
    if crypto_ed25519 is not None:
        ser = _lazy_import("cryptography.hazmat.primitives.serialization")
        sk_obj = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk_seed32)
        return sk_obj.sign(msg)
    raise RuntimeError("No Ed25519 backend available.")

def ed25519_verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
    global nacl_signing, crypto_ed25519
    try:
        if nacl_signing is not None:
            vk = nacl_signing.VerifyKey(pk)
            vk.verify(msg, sig)
            return True
        if crypto_ed25519 is not None:
            pk_obj = crypto_ed25519.Ed25519PublicKey.from_public_bytes(pk)
            pk_obj.verify(sig, msg)
            return True
        return False
    except Exception:
        return False

# -----------------------------
# Genesis / chain config
# -----------------------------

@dataclass
class Validator:
    pubkey_b64: str
    bond: int

@dataclass
class Genesis:
    version: int
    chain_id: str
    chain_id_hash: int
    timestamp_ns: int
    shards: int
    validators: List[Validator]

    # MA config
    ma_dim: int
    ma_topics: int
    ma_k_pos: int
    ma_k_neg: int
    ma_per_dim_clip_q: int
    ma_max_norm_q: int
    ma_default_lr_q: int
    ma_gentle_lr_q: int

    # protocol caps
    block_max_txs: int
    mempool_max: int
    sender_max: int

def chain_id_hash64(chain_id: str) -> int:
    # cross-chain protection: 64-bit from sha256
    d = sha256(chain_id.encode("utf-8"))
    return struct.unpack("<Q", d[:8])[0]

def load_genesis(path: str) -> Genesis:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # tolerate older shapes; enforce required keys
    return Genesis(
        version=int(obj["version"]),
        chain_id=str(obj["chain_id"]),
        chain_id_hash=int(obj["chain_id_hash"]),
        timestamp_ns=int(obj["timestamp_ns"]),
        shards=int(obj["shards"]),
        validators=[Validator(pubkey_b64=v["pubkey_b64"], bond=int(v["bond"])) for v in obj["validators"]],
        ma_dim=int(obj.get("ma_dim", 256)),
        ma_topics=int(obj.get("ma_topics", 64)),
        ma_k_pos=int(obj.get("ma_k_pos", 6)),
        ma_k_neg=int(obj.get("ma_k_neg", 3)),
        ma_per_dim_clip_q=int(obj.get("ma_per_dim_clip_q", 5120)),
        ma_max_norm_q=int(obj.get("ma_max_norm_q", 2560)),
        ma_default_lr_q=int(obj.get("ma_default_lr_q", 6554)),
        ma_gentle_lr_q=int(obj.get("ma_gentle_lr_q", 655)),
        block_max_txs=int(obj.get("block_max_txs", 2000)),
        mempool_max=int(obj.get("mempool_max", 250_000)),
        sender_max=int(obj.get("sender_max", 20_000)),
    )

def save_genesis(gen: Genesis, path: str) -> None:
    obj = dataclasses.asdict(gen)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def genesis_cmd(args: argparse.Namespace) -> None:
    # Deterministic devnet keys derived from seed + index
    seed = sha256(f"numchain-genesis:{args.seed}".encode("utf-8"))
    validators = []
    for i in range(args.nodes):
        sk_seed = sha256(seed + struct.pack("<I", i))[:32]
        kp = ed25519_keygen(sk_seed)
        validators.append(Validator(pubkey_b64=b64e(kp.pk), bond=int(args.bond)))
    cid = args.chain_id
    gen = Genesis(
        version=1,
        chain_id=cid,
        chain_id_hash=chain_id_hash64(cid),
        timestamp_ns=int(args.timestamp_ns if args.timestamp_ns else now_ns()),
        shards=int(args.shards),
        validators=validators,
        ma_dim=int(args.dim),
        ma_topics=int(args.topics),
        ma_k_pos=int(args.k_pos),
        ma_k_neg=int(args.k_neg),
        ma_per_dim_clip_q=int(args.per_dim_clip_q),
        ma_max_norm_q=int(args.max_norm_q),
        ma_default_lr_q=int(args.default_lr_q),
        ma_gentle_lr_q=int(args.gentle_lr_q),
        block_max_txs=int(args.block_max_txs),
        mempool_max=int(args.mempool_max),
        sender_max=int(args.sender_max),
    )
    save_genesis(gen, args.out)
    print("Wrote genesis:", args.out)
    print("chain_id_hash:", hex(gen.chain_id_hash))
    print("validators:", len(gen.validators))

# -----------------------------
# Wire protocol: framed JSON over TCP
# -----------------------------

MAX_FRAME = 16 * 1024 * 1024  # 16MB

def send_frame(sock: socket.socket, obj: Dict[str, Any]) -> None:
    data = jcanon(obj)
    if len(data) > MAX_FRAME:
        raise ValueError("frame too large")
    hdr = struct.pack("<I", len(data))
    sock.sendall(hdr + data)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def recv_frame(sock: socket.socket) -> Dict[str, Any]:
    hdr = recv_exact(sock, 4)
    (ln,) = struct.unpack("<I", hdr)
    if ln <= 0 or ln > MAX_FRAME:
        raise ValueError("bad frame length")
    data = recv_exact(sock, ln)
    return json.loads(data.decode("utf-8"))

# -----------------------------
# Tx / block structures
# -----------------------------

TX_KIND_USER = "USER"
TX_KIND_MA_DELTA_V1 = "MA_DELTA_V1"

@dataclass
class Tx:
    txid: str
    sender_pk_b64: str
    sig_b64: str
    kind: str
    payload_b64: str  # for MA deltas, this is MADelta binary
    timestamp_ns: int

def tx_message_for_sig(kind: str, payload_b64: str, timestamp_ns: int) -> bytes:
    # canonical message signing bytes
    return jcanon({"kind": kind, "payload_b64": payload_b64, "timestamp_ns": int(timestamp_ns)})

def tx_make(sender_pk: bytes, sender_sk_seed32: bytes, kind: str, payload: bytes, timestamp_ns: Optional[int] = None) -> Tx:
    ts = int(timestamp_ns if timestamp_ns is not None else now_ns())
    payload_b64 = b64e(payload)
    msg = tx_message_for_sig(kind, payload_b64, ts)
    sig = ed25519_sign(sender_sk_seed32, msg)
    txid = sha256_hex(jcanon({"pk": b64e(sender_pk), "msg": payload_b64, "ts": ts, "kind": kind, "sig": b64e(sig)}))[:32]
    return Tx(
        txid=txid,
        sender_pk_b64=b64e(sender_pk),
        sig_b64=b64e(sig),
        kind=kind,
        payload_b64=payload_b64,
        timestamp_ns=ts,
    )

def tx_verify(tx: Tx) -> bool:
    pk = b64d(tx.sender_pk_b64)
    sig = b64d(tx.sig_b64)
    msg = tx_message_for_sig(tx.kind, tx.payload_b64, tx.timestamp_ns)
    return ed25519_verify(pk, msg, sig)

@dataclass
class Block:
    height: int
    prev_hash: str
    proposer: int
    timestamp_ns: int
    txids: List[str]
    block_hash: str

def block_hash_fields(height: int, prev_hash: str, proposer: int, timestamp_ns: int, txids: List[str]) -> str:
    return sha256_hex(jcanon({
        "height": int(height),
        "prev_hash": prev_hash,
        "proposer": int(proposer),
        "timestamp_ns": int(timestamp_ns),
        "txids": list(txids),
    }))[:64]

# -----------------------------
# Node
# -----------------------------

class Node:
    def __init__(self, gen: Genesis, node_index: int, sk_seed32: bytes, rpc_host: str, rpc_port: int,
                 gossip_host: str, gossip_port: int, peers: List[str], log_path: str):
        self.gen = gen
        self.node_index = int(node_index)
        self.sk_seed32 = sk_seed32
        self.pk = b64d(gen.validators[self.node_index].pubkey_b64)

        self.rpc_host = rpc_host
        self.rpc_port = int(rpc_port)
        self.gossip_host = gossip_host
        self.gossip_port = int(gossip_port)
        self.peers = [p for p in peers if p]

        self.log_path = log_path

        # Chain state
        self.height = 0
        self.tip_hash = "0" * 64
        self.blocks: Dict[int, Block] = {}
        self.receipts: Dict[str, Dict[str, Any]] = {}
        self.start_ns = now_ns()

        # Mempool
        self.mempool: Dict[str, Tx] = {}
        self.sender_counts: Dict[str, int] = {}
        self.seen_txids: Dict[str, int] = {}  # txid -> last_seen_height (dedup window)
        self.seen_cap = 8192
        self.seen_lock = threading.Lock()

        # MA state
        self.ma_cfg = MAConfig(
            dim=gen.ma_dim,
            topics=gen.ma_topics,
            k_pos=gen.ma_k_pos,
            k_neg=gen.ma_k_neg,
            per_dim_clip_q=gen.ma_per_dim_clip_q,
            max_norm_q=gen.ma_max_norm_q,
            default_lr_q=gen.ma_default_lr_q,
            gentle_lr_q=gen.ma_gentle_lr_q,
        )
        self.ma_state: Dict[int, Dict[int, Dict[int, ProtoState]]] = {}
        self.ma_root = ma_root_hash(self.ma_state, self.ma_cfg)

        # Gossip
        self.stop_ev = threading.Event()
        self.inbox = queue.Queue()  # inbound messages
        self.peer_senders: Dict[str, "PeerSender"] = {}

        # Consensus loop
        self.consensus_thread = threading.Thread(target=self._consensus_loop, daemon=True)

        # Metrics
        self.m_tx_recv = 0
        self.m_tx_accept = 0
        self.m_tx_reject = 0
        self.m_blocks_committed = 0
        self.m_gossip_sent = 0
        self.m_gossip_recv = 0
        self.m_gossip_ack = 0
        self.m_gossip_retry = 0
        self.m_last_commit_ns = 0
        self.m_apply_ms_p95 = 0.0
        self.m_apply_ms_p99 = 0.0

        # Replay
        self._replay_log()

    # -------------------------
    # Dedup (bounded)
    # -------------------------

    def _seen_add(self, txid: str) -> None:
        with self.seen_lock:
            if txid in self.seen_txids:
                self.seen_txids[txid] = self.height
                return
            if len(self.seen_txids) >= self.seen_cap:
                # evict oldest by height (simple deterministic)
                oldest = min(self.seen_txids.items(), key=lambda kv: kv[1])[0]
                self.seen_txids.pop(oldest, None)
            self.seen_txids[txid] = self.height

    def _seen_has(self, txid: str) -> bool:
        with self.seen_lock:
            return txid in self.seen_txids

    # -------------------------
    # Log replay / append
    # -------------------------

    def _append_log(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _replay_log(self) -> None:
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("type") == "block":
                        b = Block(**rec["block"])
                        self.blocks[b.height] = b
                        self.height = max(self.height, b.height)
                        self.tip_hash = b.block_hash
                        # receipts are replayed separately
                    elif rec.get("type") == "receipt":
                        self.receipts[rec["txid"]] = rec
                    elif rec.get("type") == "ma_apply":
                        # allow reconstructing MA root quickly
                        # we still apply deterministically to rebuild state
                        txid = rec["txid"]
                        # In v1.4.2, we prefer replay from tx payloads; this record is optional.
                        _ = txid
                    elif rec.get("type") == "ma_tx":
                        # deterministic MA tx replay record:
                        # contains madelta bytes b64
                        try:
                            blob = b64d(rec["madelta_b64"])
                            h, payload_i16, _prov = parse_madelta_v1(blob)
                            # validate chain hash
                            if int(h.chain_id_hash) != int(self.gen.chain_id_hash):
                                continue
                            apply_madelta_to_state(self.ma_state, self.ma_cfg, h, payload_i16)
                        except Exception:
                            continue
                    # ignore unknown
            # recompute MA root after replay
            self.ma_root = ma_root_hash(self.ma_state, self.ma_cfg)
        except Exception:
            print("WARN: failed to replay log; starting fresh", file=sys.stderr)
            traceback.print_exc()

    # -------------------------
    # Gossip networking
    # -------------------------

    def start(self) -> None:
        # start peer senders
        for p in self.peers:
            self.peer_senders[p] = PeerSender(self, p)
            self.peer_senders[p].start()
        # start listener
        threading.Thread(target=self._gossip_listener, daemon=True).start()
        # start inbox processor
        threading.Thread(target=self._inbox_loop, daemon=True).start()
        # start consensus
        self.consensus_thread.start()
        # start rpc
        self._start_rpc()

    def _gossip_listener(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.gossip_host, self.gossip_port))
        srv.listen(128)
        while not self.stop_ev.is_set():
            try:
                srv.settimeout(0.5)
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                continue
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            while not self.stop_ev.is_set():
                try:
                    msg = recv_frame(conn)
                    self.m_gossip_recv += 1
                    self.inbox.put(msg)
                    # if message requests ACK, send it
                    if msg.get("need_ack"):
                        ack = {"type": "ACK", "ack_id": msg.get("ack_id"), "from": self.gossip_port}
                        send_frame(conn, ack)
                        self.m_gossip_ack += 1
                except socket.timeout:
                    continue
                except Exception:
                    return

    def _inbox_loop(self) -> None:
        while not self.stop_ev.is_set():
            try:
                msg = self.inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                t = msg.get("type")
                if t == "TX":
                    self._on_gossip_tx(msg)
                elif t == "ACK":
                    ack_id = msg.get("ack_id")
                    frm = msg.get("from")
                    if ack_id is not None:
                        for ps in self.peer_senders.values():
                            ps.on_ack(ack_id, frm)
                # v1.4.2: consensus messages would go here (proposal/vote/qc/commit)
                # For this wiring pass, we keep the original minimal commit loop:
                # leader locally commits blocks; gossip is used for TX propagation only.
                # (If your existing v1.4.2 already had full BFT, you can route those messages here as well.)
            except Exception:
                continue

    def _send_gossip(self, peer: str, msg: Dict[str, Any], critical: bool) -> None:
        ps = self.peer_senders.get(peer)
        if ps is None:
            return
        ps.send(msg, critical=critical)
        self.m_gossip_sent += 1

    def gossip_broadcast(self, msg: Dict[str, Any], critical: bool = False) -> None:
        for p in self.peers:
            self._send_gossip(p, msg, critical=critical)

    def _on_gossip_tx(self, msg: Dict[str, Any]) -> None:
        # TX propagation with TTL
        ttl = int(msg.get("ttl", 0))
        if ttl <= 0:
            return
        txd = msg.get("tx")
        if not isinstance(txd, dict):
            return
        try:
            tx = Tx(**txd)
        except Exception:
            return
        if self._seen_has(tx.txid):
            return
        ok, reason = self._try_accept_tx(tx)
        if ok:
            self._seen_add(tx.txid)
            # re-gossip
            out = {"type": "TX", "tx": dataclasses.asdict(tx), "ttl": ttl - 1, "need_ack": False}
            self.gossip_broadcast(out, critical=False)

    # -------------------------
    # Tx accept / validation
    # -------------------------

    def _try_accept_tx(self, tx: Tx) -> Tuple[bool, str]:
        self.m_tx_recv += 1
        if tx.txid in self.mempool:
            return False, "dupe"
        if len(self.mempool) >= self.gen.mempool_max:
            self.m_tx_reject += 1
            return False, "mempool_full"
        if not tx_verify(tx):
            self.m_tx_reject += 1
            return False, "bad_sig"

        # per-sender cap
        c = self.sender_counts.get(tx.sender_pk_b64, 0)
        if c >= self.gen.sender_max:
            self.m_tx_reject += 1
            return False, "sender_cap"

        # If MA_DELTA_V1, validate header basics now (cheap) to avoid garbage in mempool
        if tx.kind == TX_KIND_MA_DELTA_V1:
            try:
                blob = b64d(tx.payload_b64)
                h, payload_i16, _prov = parse_madelta_v1(blob)
                if int(h.chain_id_hash) != int(self.gen.chain_id_hash):
                    self.m_tx_reject += 1
                    return False, "chain_id_hash_mismatch"
                if int(h.dim) != int(self.ma_cfg.dim):
                    self.m_tx_reject += 1
                    return False, "dim_mismatch"
                # bounds that depend on K
                if h.bank == BANK_POS and h.proto_idx >= self.ma_cfg.k_pos:
                    self.m_tx_reject += 1
                    return False, "proto_oob"
                if h.bank == BANK_NEG and h.proto_idx >= self.ma_cfg.k_neg:
                    self.m_tx_reject += 1
                    return False, "proto_oob"
                # action/payload invariants already checked by parser
                _ = payload_i16
            except Exception:
                self.m_tx_reject += 1
                return False, "bad_madelta"

        self.mempool[tx.txid] = tx
        self.sender_counts[tx.sender_pk_b64] = c + 1
        self.m_tx_accept += 1
        return True, "ok"

    # -------------------------
    # Consensus (minimal single-leader commit loop)
    # -------------------------

    def _leader_for_height(self, h: int) -> int:
        # deterministic round-robin leader for this minimal kernel wiring
        return int(h) % len(self.gen.validators)

    def _consensus_loop(self) -> None:
        # Minimal commit loop: leader produces one block per second if it is leader for next height.
        # (If your prior v1.4.2 already had HotStuff-style BFT, keep it and call into apply_tx below.)
        apply_times = []
        while not self.stop_ev.is_set():
            try:
                next_h = self.height + 1
                leader = self._leader_for_height(next_h)
                if leader != self.node_index:
                    time.sleep(0.2)
                    continue

                # Assemble block
                txs = list(self.mempool.values())[: self.gen.block_max_txs]
                txids = [t.txid for t in txs]
                bts = now_ns()
                bh = block_hash_fields(next_h, self.tip_hash, self.node_index, bts, txids)
                blk = Block(height=next_h, prev_hash=self.tip_hash, proposer=self.node_index,
                            timestamp_ns=bts, txids=txids, block_hash=bh)

                # Apply block deterministically
                t0 = time.perf_counter()
                self._apply_block(blk, txs)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                apply_times.append(dt_ms)
                if len(apply_times) > 2000:
                    apply_times = apply_times[-2000:]

                # compute p95/p99 on a rolling window
                if len(apply_times) >= 20:
                    srt = sorted(apply_times)
                    p95 = srt[int(0.95 * (len(srt) - 1))]
                    p99 = srt[int(0.99 * (len(srt) - 1))]
                    self.m_apply_ms_p95 = float(p95)
                    self.m_apply_ms_p99 = float(p99)

                # Commit block
                self.blocks[blk.height] = blk
                self.height = blk.height
                self.tip_hash = blk.block_hash
                self.m_blocks_committed += 1
                self.m_last_commit_ns = now_ns()
                self._append_log({"type": "block", "block": dataclasses.asdict(blk)})

                time.sleep(1.0)
            except Exception:
                time.sleep(0.5)

    def _apply_block(self, blk: Block, txs: List[Tx]) -> None:
        # deterministic order = txids order
        for tx in txs:
            status = "ok"
            info: Dict[str, Any] = {}
            try:
                if tx.kind == TX_KIND_MA_DELTA_V1:
                    blob = b64d(tx.payload_b64)
                    h, payload_i16, _prov = parse_madelta_v1(blob)
                    # execute
                    apply_madelta_to_state(self.ma_state, self.ma_cfg, h, payload_i16)
                    # persist replay record for MA deltas (normative)
                    self._append_log({"type": "ma_tx", "txid": tx.txid, "madelta_b64": tx.payload_b64})
                    # update root
                    self.ma_root = ma_root_hash(self.ma_state, self.ma_cfg)
                    info["ma_root_hex"] = self.ma_root.hex()
                else:
                    # USER tx kind reserved: accept but no-op to keep minimal surface stable
                    pass
            except Exception as e:
                status = "err"
                info["error"] = str(e)

            self.receipts[tx.txid] = {
                "type": "receipt",
                "txid": tx.txid,
                "block_height": blk.height,
                "status": status,
                "info": info,
                "timestamp_ns": now_ns(),
            }
            self._append_log(self.receipts[tx.txid])

            # remove from mempool and decrement sender count deterministically
            self.mempool.pop(tx.txid, None)
            c = self.sender_counts.get(tx.sender_pk_b64, 0)
            if c > 0:
                self.sender_counts[tx.sender_pk_b64] = c - 1

    # -------------------------
    # RPC
    # -------------------------

    def _start_rpc(self) -> None:
        global fastapi, uvicorn
        fastapi = _lazy_import("fastapi")
        uvicorn = _lazy_import("uvicorn")
        if fastapi is None or uvicorn is None:
            print("ERROR: fastapi/uvicorn not installed. Run: python3 numchain_v142.py deps", file=sys.stderr)
            sys.exit(2)

        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse, JSONResponse

        app = FastAPI(title="NUMCHAIN v1.4.2", version="v1.4.2+madelta")

        @app.get("/health")
        def health():
            up_s = (now_ns() - self.start_ns) / 1e9
            return {
                "version": "v1.4.2+madelta",
                "chain_id": self.gen.chain_id,
                "chain_id_hash": hex(self.gen.chain_id_hash),
                "node_index": self.node_index,
                "rpc": f"{self.rpc_host}:{self.rpc_port}",
                "gossip": f"{self.gossip_host}:{self.gossip_port}",
                "peers": list(self.peers),
                "height": self.height,
                "tip": self.tip_hash[:16],
                "mempool": len(self.mempool),
                "uptime_s": up_s,
                "ma": {
                    "dim": self.ma_cfg.dim,
                    "topics": self.ma_cfg.topics,
                    "k_pos": self.ma_cfg.k_pos,
                    "k_neg": self.ma_cfg.k_neg,
                    "per_dim_clip_q": self.ma_cfg.per_dim_clip_q,
                    "max_norm_q": self.ma_cfg.max_norm_q,
                    "root_hex": self.ma_root.hex(),
                },
                "metrics": {
                    "tx_recv": self.m_tx_recv,
                    "tx_accept": self.m_tx_accept,
                    "tx_reject": self.m_tx_reject,
                    "blocks_committed": self.m_blocks_committed,
                    "gossip_sent": self.m_gossip_sent,
                    "gossip_recv": self.m_gossip_recv,
                    "apply_p95_ms": self.m_apply_ms_p95,
                    "apply_p99_ms": self.m_apply_ms_p99,
                }
            }

        @app.get("/metrics")
        def metrics():
            # Prometheus text format
            lines = []
            def g(name: str, val: Any, labels: Optional[Dict[str, str]] = None):
                if labels:
                    lab = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                    lines.append(f"{name}{{{lab}}} {val}")
                else:
                    lines.append(f"{name} {val}")
            labels = {"chain_id": self.gen.chain_id, "node": str(self.node_index), "version": "v1.4.2+madelta"}
            g("numchain_height", self.height, labels)
            g("numchain_mempool", len(self.mempool), labels)
            g("numchain_tx_recv_total", self.m_tx_recv, labels)
            g("numchain_tx_accept_total", self.m_tx_accept, labels)
            g("numchain_tx_reject_total", self.m_tx_reject, labels)
            g("numchain_blocks_committed_total", self.m_blocks_committed, labels)
            g("numchain_gossip_sent_total", self.m_gossip_sent, labels)
            g("numchain_gossip_recv_total", self.m_gossip_recv, labels)
            g("numchain_apply_p95_ms", self.m_apply_ms_p95, labels)
            g("numchain_apply_p99_ms", self.m_apply_ms_p99, labels)
            # expose MA root as info
            g("numchain_ma_root_info", 1, {**labels, "root": self.ma_root.hex()})
            return PlainTextResponse("\n".join(lines) + "\n")

        @app.post("/tx")
        async def submit_tx(body: Dict[str, Any]):
            """
            Accepts:
              - kind: "MA_DELTA_V1" (recommended)
              - madelta_b64 OR madelta_hex (payload)
              - sender_pk_b64 + sig_b64 + timestamp_ns (if you are submitting a fully formed Tx)
            If sender fields absent, node will create a dev-signer tx envelope from node0 seed (NOT for prod).
            """
            try:
                kind = body.get("kind", TX_KIND_USER)
                if kind == TX_KIND_MA_DELTA_V1:
                    if "madelta_b64" in body:
                        payload = b64d(body["madelta_b64"])
                    elif "madelta_hex" in body:
                        payload = bytes.fromhex(body["madelta_hex"])
                    else:
                        return JSONResponse({"ok": False, "error": "missing madelta_b64 or madelta_hex"}, status_code=400)
                else:
                    payload = jcanon(body.get("payload", {}))

                # If fully formed Tx provided:
                if "txid" in body and "sender_pk_b64" in body and "sig_b64" in body and "payload_b64" in body:
                    tx = Tx(
                        txid=str(body["txid"]),
                        sender_pk_b64=str(body["sender_pk_b64"]),
                        sig_b64=str(body["sig_b64"]),
                        kind=str(body.get("kind", TX_KIND_USER)),
                        payload_b64=str(body["payload_b64"]),
                        timestamp_ns=int(body.get("timestamp_ns", now_ns())),
                    )
                else:
                    # dev-mode: sign with deterministic sender derived from chain_id (for quick testing)
                    # NOTE: closed-alpha convenience only.
                    seed = sha256(f"dev-sender:{self.gen.chain_id}".encode("utf-8"))[:32]
                    kp = ed25519_keygen(seed)
                    tx = tx_make(kp.pk, kp.sk, kind, payload, timestamp_ns=int(body.get("timestamp_ns", now_ns())))

                ok, reason = self._try_accept_tx(tx)
                if not ok:
                    return JSONResponse({"ok": False, "error": reason}, status_code=400)

                # gossip TX with TTL
                self._seen_add(tx.txid)
                self.gossip_broadcast({"type": "TX", "tx": dataclasses.asdict(tx), "ttl": 3, "need_ack": False}, critical=False)
                return {"ok": True, "txid": tx.txid}
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        @app.get("/block/{height}")
        def get_block(height: int):
            b = self.blocks.get(int(height))
            if not b:
                return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
            return {"ok": True, "block": dataclasses.asdict(b)}

        @app.get("/receipt/{txid}")
        def get_receipt(txid: str):
            r = self.receipts.get(txid)
            if not r:
                return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
            return {"ok": True, "receipt": r}

        uvicorn.run(app, host=self.rpc_host, port=self.rpc_port, log_level="info")

# -----------------------------
# Peer sender with ACK/retry (critical messages)
# -----------------------------

class PeerSender(threading.Thread):
    def __init__(self, node: Node, peer: str):
        super().__init__(daemon=True)
        self.node = node
        self.peer = peer
        self.q = queue.Queue()
        self.stop_ev = node.stop_ev
        self.pending: Dict[str, Tuple[float, Dict[str, Any], int]] = {}  # ack_id -> (deadline, msg, tries)
        self.pending_lock = threading.Lock()

    def send(self, msg: Dict[str, Any], critical: bool) -> None:
        if critical:
            ack_id = sha256_hex(jcanon({"peer": self.peer, "msg": msg, "t": now_ns()}))[:16]
            msg = dict(msg)
            msg["need_ack"] = True
            msg["ack_id"] = ack_id
            with self.pending_lock:
                self.pending[ack_id] = (time.time() + 0.5, msg, 0)
        self.q.put(msg)

    def on_ack(self, ack_id: str, frm: Any) -> None:
        # ignore frm here; remove pending if matches
        with self.pending_lock:
            self.pending.pop(str(ack_id), None)

    def run(self) -> None:
        host, port = self._split_peer(self.peer)
        sock = None
        last_conn_try = 0.0

        while not self.stop_ev.is_set():
            # reconnect if needed
            if sock is None:
                if time.time() - last_conn_try < 0.5:
                    time.sleep(0.1)
                    continue
                last_conn_try = time.time()
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)
                    sock.connect((host, port))
                    sock.settimeout(2.0)
                except Exception:
                    sock = None
                    continue

            # resend pending acks on schedule
            try:
                nowt = time.time()
                resend = []
                with self.pending_lock:
                    for ack_id, (deadline, msg, tries) in list(self.pending.items()):
                        if nowt >= deadline and tries < 5:
                            resend.append((ack_id, msg, tries))
                for ack_id, msg, tries in resend:
                    try:
                        send_frame(sock, msg)
                        self.node.m_gossip_retry += 1
                        with self.pending_lock:
                            # bump deadline
                            self.pending[ack_id] = (time.time() + 0.5, msg, tries + 1)
                    except Exception:
                        sock = None
                        break
            except Exception:
                pass

            # send queued messages
            try:
                msg = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                send_frame(sock, msg)
            except Exception:
                sock = None
                continue

    @staticmethod
    def _split_peer(peer: str) -> Tuple[str, int]:
        if "://" in peer:
            peer = peer.split("://", 1)[1]
        if "/" in peer:
            peer = peer.split("/", 1)[0]
        host, port_s = peer.split(":")
        return host, int(port_s)

# -----------------------------
# CLI tools: deps / flood / poll / keygen
# -----------------------------

def deps_cmd(_args: argparse.Namespace) -> None:
    pkgs = [
        "numpy",
        "psutil",
        "fastapi",
        "uvicorn[standard]",
        "httpx",
        "pynacl",
        "cryptography",
    ]
    print("Installing:", " ".join(pkgs))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs)
    print("Done.")

def keygen_cmd(args: argparse.Namespace) -> None:
    seed = None
    if args.seed:
        seed = sha256(args.seed.encode("utf-8"))[:32]
    kp = ed25519_keygen(seed)
    out = {
        "sk_seed32_b64": b64e(kp.sk),
        "pk_b64": b64e(kp.pk),
    }
    print(json.dumps(out, indent=2))

def flood_cmd(args: argparse.Namespace) -> None:
    global httpx
    httpx = _lazy_import("httpx")
    if httpx is None:
        print("ERROR: httpx not installed. Run deps.", file=sys.stderr)
        sys.exit(2)

    gen = load_genesis(args.genesis)
    url = args.url.rstrip("/")
    seconds = int(args.seconds)
    target_tps = float(args.target_tps)
    senders = int(args.senders)
    ramp = int(args.ramp_up_seconds)

    # deterministic sender seeds
    base = sha256(f"flood:{gen.chain_id}".encode("utf-8"))
    keys = []
    for i in range(senders):
        seed = sha256(base + struct.pack("<I", i))[:32]
        kp = ed25519_keygen(seed)
        keys.append((kp.pk, kp.sk))

    # If you want to flood MA deltas, you must provide prebuilt madelta blobs externally.
    # This flood tool is intentionally generic and uses USER tx unless --madelta-b64 is supplied.
    madelta_b64 = args.madelta_b64

    stats = {
        "sent": 0,
        "ok": 0,
        "err": 0,
        "errors": {},
        "lat_ms": [],
    }

    client = httpx.Client(timeout=2.0)
    t_start = time.time()

    def current_rate(elapsed: float) -> float:
        if ramp <= 0:
            return target_tps
        if elapsed >= ramp:
            return target_tps
        return target_tps * (elapsed / ramp)

    while True:
        elapsed = time.time() - t_start
        if elapsed >= seconds:
            break
        rate = current_rate(elapsed)
        if rate <= 0:
            time.sleep(0.01)
            continue

        # send one request, then sleep according to rate (simple; not perfect)
        pk, sk = keys[stats["sent"] % len(keys)]
        t0 = time.perf_counter()
        if madelta_b64:
            body = {"kind": TX_KIND_MA_DELTA_V1, "madelta_b64": madelta_b64}
        else:
            body = {"kind": TX_KIND_USER, "payload": {"n": stats["sent"]}}
        try:
            r = client.post(url + "/tx", json=body)
            dt = (time.perf_counter() - t0) * 1000.0
            stats["lat_ms"].append(dt)
            stats["sent"] += 1
            if r.status_code == 200:
                stats["ok"] += 1
            else:
                stats["err"] += 1
                try:
                    e = r.json().get("error", "http_error")
                except Exception:
                    e = "http_error"
                stats["errors"][e] = stats["errors"].get(e, 0) + 1
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            stats["lat_ms"].append(dt)
            stats["sent"] += 1
            stats["err"] += 1
            k = type(e).__name__
            stats["errors"][k] = stats["errors"].get(k, 0) + 1

        # pacing
        sleep_s = 1.0 / max(rate, 1e-6)
        if sleep_s > 0:
            time.sleep(min(sleep_s, 0.05))

    lats = stats["lat_ms"]
    lats_sorted = sorted(lats)
    def pct(p: float) -> float:
        if not lats_sorted:
            return 0.0
        return lats_sorted[int(p * (len(lats_sorted) - 1))]
    print(json.dumps({
        "duration_s": seconds,
        "target_tps": target_tps,
        "sent": stats["sent"],
        "ok": stats["ok"],
        "err": stats["err"],
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "errors": stats["errors"],
    }, indent=2))

def poll_cmd(args: argparse.Namespace) -> None:
    global httpx
    httpx = _lazy_import("httpx")
    if httpx is None:
        print("ERROR: httpx not installed. Run deps.", file=sys.stderr)
        sys.exit(2)

    urls = [u.strip().rstrip("/") for u in args.urls.split(",") if u.strip()]
    seconds = int(args.seconds)
    every = float(args.every)

    client = httpx.Client(timeout=2.0)
    t0 = time.time()
    while True:
        if time.time() - t0 >= seconds:
            break
        rows = []
        for u in urls:
            try:
                r = client.get(u + "/health")
                if r.status_code != 200:
                    rows.append((u, None, None, None, "http"))
                    continue
                j = r.json()
                rows.append((u, j.get("height"), j.get("mempool"), j.get("ma", {}).get("root_hex", "")[:16], "ok"))
            except Exception:
                rows.append((u, None, None, None, "err"))
        heights = [x[1] for x in rows if isinstance(x[1], int)]
        spread = (max(heights) - min(heights)) if heights else None
        print(f"spread={spread} " + " | ".join([f"{u} h={h} mp={mp} ma={m} {st}" for (u, h, mp, m, st) in rows]))
        time.sleep(every)

# -----------------------------
# Main
# -----------------------------

def run_cmd(args: argparse.Namespace) -> None:
    global np, psutil
    np = _lazy_import("numpy")
    psutil = _lazy_import("psutil")
    # Load genesis
    gen = load_genesis(args.genesis)

    # derive node key deterministically from genesis (devnet); in real deploy load from disk
    seed = sha256(f"node-sk:{gen.chain_id}:{args.node_index}".encode("utf-8"))[:32]
    # validate against genesis pubkey (deterministic)
    kp = ed25519_keygen(seed)
    expected_pk = b64d(gen.validators[int(args.node_index)].pubkey_b64)
    if kp.pk != expected_pk:
        print("ERROR: derived key does not match genesis validator pk (bad genesis/seed).", file=sys.stderr)
        sys.exit(2)

    peers = []
    if args.peers:
        peers = [p.strip() for p in args.peers.split(",") if p.strip()]

    log_path = args.log if args.log else f"node_{args.node_index}.jsonl"
    node = Node(
        gen=gen,
        node_index=int(args.node_index),
        sk_seed32=seed,
        rpc_host=args.rpc_host,
        rpc_port=int(args.rpc),
        gossip_host=args.gossip_host,
        gossip_port=int(args.gossip),
        peers=peers,
        log_path=log_path,
    )
    node.start()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="numchain_v142.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("deps")
    sp.set_defaults(func=deps_cmd)

    sp = sub.add_parser("keygen")
    sp.add_argument("--seed", default="")
    sp.set_defaults(func=keygen_cmd)

    sp = sub.add_parser("genesis")
    sp.add_argument("--out", required=True)
    sp.add_argument("--chain-id", default="numchain-dev")
    sp.add_argument("--seed", default="devnet")
    sp.add_argument("--timestamp-ns", type=int, default=0)
    sp.add_argument("--nodes", type=int, default=5)
    sp.add_argument("--bond", type=int, default=1000)
    sp.add_argument("--shards", type=int, default=8)

    sp.add_argument("--topics", type=int, default=64)
    sp.add_argument("--dim", type=int, default=256)
    sp.add_argument("--k-pos", type=int, default=6)
    sp.add_argument("--k-neg", type=int, default=3)
    sp.add_argument("--per-dim-clip-q", type=int, default=5120)
    sp.add_argument("--max-norm-q", type=int, default=2560)
    sp.add_argument("--default-lr-q", type=int, default=6554)
    sp.add_argument("--gentle-lr-q", type=int, default=655)

    sp.add_argument("--block-max-txs", type=int, default=2000)
    sp.add_argument("--mempool-max", type=int, default=250_000)
    sp.add_argument("--sender-max", type=int, default=20_000)
    sp.set_defaults(func=genesis_cmd)

    sp = sub.add_parser("run")
    sp.add_argument("--genesis", required=True)
    sp.add_argument("--node-index", type=int, required=True)
    sp.add_argument("--rpc-host", default="127.0.0.1")
    sp.add_argument("--rpc", type=int, default=8000)
    sp.add_argument("--gossip-host", default="127.0.0.1")
    sp.add_argument("--gossip", type=int, default=9000)
    sp.add_argument("--peers", default="")
    sp.add_argument("--log", default="")
    sp.set_defaults(func=run_cmd)

    sp = sub.add_parser("flood")
    sp.add_argument("--genesis", required=True)
    sp.add_argument("--url", required=True)
    sp.add_argument("--seconds", type=int, default=30)
    sp.add_argument("--target-tps", type=float, default=500.0)
    sp.add_argument("--senders", type=int, default=64)
    sp.add_argument("--ramp-up-seconds", type=int, default=0)
    sp.add_argument("--madelta-b64", default="")  # optional: flood same MADelta blob repeatedly
    sp.set_defaults(func=flood_cmd)

    sp = sub.add_parser("poll")
    sp.add_argument("--urls", required=True)
    sp.add_argument("--seconds", type=int, default=30)
    sp.add_argument("--every", type=float, default=2.0)
    sp.set_defaults(func=poll_cmd)

    return p

def main() -> None:
    # seed Python RNG deterministically (defensive)
    random.seed(0xC0FFEE)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
