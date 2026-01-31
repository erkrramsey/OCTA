#!/usr/bin/env python3
# ======================================================================================
# MA_DELTA_V1.6 — REFERENCE KERNEL (SPEC-IN-CODE) + INVARIANTS + DEMOS + EVIDENCE RULES
# ======================================================================================
# What this is
#   A single-file, zero-dependency reference implementation for:
#     - A 256-dim shared state vector (Q16.16)
#     - Deterministic Perfect-T “attractor” updates (MA_DELTA)
#     - Merkle commitment (MACv1) + per-index proofs
#     - Receipt hash chain (tamper-evident history)
#     - Snapshot-accelerated replay verification
#     - Safe sync (pull_full / push_full with full validation + apply)
#     - Fork-choice + reconcile (adopt best verified chain)
#     - Policy enforcement: allowlist + M-of-N threshold signatures
#     - Evidence commitment AND OPTIONAL evidence validation rules
#
# Design invariants (v1.x)
#   1) Deterministic apply: same txlog => same post_root bit-for-bit.
#   2) Domain separation: every hash namespace is explicitly separated.
#   3) Receipts chain: each receipt commits to prev_receipt_hash (immutability).
#   4) Txid anti-malleability: txid binds payload + canonical signature pack.
#   5) Replay equivalence: replay-from-genesis == replay-from-snapshot == DB head.
#   6) Safe sync: push_full must validate+apply deterministically against local state.
#
# v1.6 upgrades (over v1.5)
#   A) Teachability:
#       - Design invariants block (above)
#       - Two runnable demo commands:
#            * demo-router-3of5: 3-of-5 approval updates a shared router vector
#            * demo-evidence-oracle: evidence hash required + oracle signer rule
#   B) Evidence validation rules (optional, policy-driven):
#       - policy.json may include evidence_rules:
#            [
#              {"evidence_type": 7, "min_signers": 1, "required_signers": ["<pubkeyhex>", ...]},
#              {"evidence_type": 1, "min_signers": 2, "required_signers": []}  # empty => use allowlist
#            ]
#       - If a tx includes evidence_type matching a rule:
#            * evidence MUST be present
#            * valid_signers must satisfy min_signers from required_signers-set
#
# Optional domain override (advanced forks)
#   You can override domain prefixes via environment variables (ASCII strings):
#     MA_DOMAIN_TX, MA_DOMAIN_MERKLE_LEAF, MA_DOMAIN_MERKLE_NODE, MA_DOMAIN_TXID,
#     MA_DOMAIN_SIGPACK, MA_DOMAIN_RCPT_HASH
#   IMPORTANT: all nodes in the same chain MUST use identical domains.
#
# RPC:
#   GET  /health
#   GET  /metrics
#   GET  /policy
#   GET  /state/root
#   GET  /state
#   GET  /receipt/head
#   GET  /receipt/{height}
#   GET  /receipt/hash/{height}
#   GET  /receipt/range?from=H&limit=N
#   GET  /evidence/{height}
#   GET  /proof/{index}
#   GET  /tx/{height}
#   GET  /tx/range?from=H&limit=N
#   POST /submit                     {tx_hex|tx_b64}
#   POST /sync/heads                 {}
#   POST /sync/pull_full             {from_height, limit}
#   POST /sync/push_full             {receipts:[...], txs:[...]}   (SAFE: validate+apply)
#   POST /admin/verify               {commit?:bool}
#   POST /admin/reconcile            {peers:[...], pull_chunk?:int}
#
# CLI:
#   python ma_delta_v1_6.py init            --dir data --chain-id 1
#   python ma_delta_v1_6.py policy-set      --dir data --chain-id 1 --threshold 2 --allowlist pubkeys.txt
#   python ma_delta_v1_6.py policy-add      --dir data --pubkey-hex <64hex>
#   python ma_delta_v1_6.py policy-add-rule --dir data --evidence-type 7 --min-signers 1 --required-signers <pk1,pk2,...>
#   python ma_delta_v1_6.py run             --dir data --chain-id 1 --host 127.0.0.1 --port 8080
#   python ma_delta_v1_6.py make-tx         --out tx.bin --chain-id 1 --sign --seed-hex <32B>
#                                          --extra-seeds <seed1hex,seed2hex,...>
#                                          --evidence-file evidence.bin --evidence-type 7
#   python ma_delta_v1_6.py submit          --url http://127.0.0.1:8080 --tx tx.bin
#   python ma_delta_v1_6.py verify          --dir data --chain-id 1 --commit
#   python ma_delta_v1_6.py reconcile       --dir data --chain-id 1 --peers http://127.0.0.1:8081,http://127.0.0.1:8082
#   python ma_delta_v1_6.py demo-router-3of5
#   python ma_delta_v1_6.py demo-evidence-oracle
#   python ma_delta_v1_6.py selftest
# ======================================================================================

from __future__ import annotations
import argparse
import base64
import dataclasses
import hashlib
import http.client
import json
import os
import re
import shutil
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple, Optional, Dict, Any

# ----------------------------- Constants / Spec --------------------------------------

MAGIC = b"MAD1"
VERSION = 1
DIM = 256

OP_PERFECT_T = 1

FLAG_CLIP = 1 << 0
FLAG_ETA_CLAMP = 1 << 1
FLAG_INCLUDE_PUBKEY = 1 << 2
FLAG_MULTI_SIG = 1 << 3
FLAG_HAS_TLV = 1 << 4

HEADER_FMT = "<4sHHBBH I Q I I Q Q I 12s"
HEADER_SIZE = 64

DEFAULT_ETA_CAP_Q0_32 = int(1.999 * (1 << 32))  # keep < 2.0
DEFAULT_EPS_Q32_32 = int(1 << 32)               # 1.0
DEFAULT_ALPHA_Q0_32 = int(0.25 * (1 << 32))     # 0.25

DEFAULT_CLIP_MIN_Q16_16 = -(1 << 30)
DEFAULT_CLIP_MAX_Q16_16 = +(1 << 30)

STATE_JSON = "state.json"
RECEIPTS_JSONL = "receipts.jsonl"
TXLOG_JSONL = "txlog.jsonl"
SNAP_DIR = "snaps"
POLICY_JSON = "policy.json"

DEFAULT_SNAPSHOT_EVERY = 50
DEFAULT_KEEP_SNAPSHOTS = 20

DEFAULT_MAX_SUBMIT_BYTES = 64 + (DIM * 2) + 4096 + 32 + 64 + 1 + (32 + 64) * 32
DEFAULT_MAX_PULL = 200
DEFAULT_MAX_RANGE = 200
DEFAULT_HTTP_TIMEOUT_S = 8

# TLV types
TLV_EVIDENCE = 0x01  # [u8 evidence_type][32 bytes evidence_hash_sha256]

# ----------------------------- Domain Separation (env overridable) -------------------

def _domain_env(name: str, default: bytes) -> bytes:
    v = os.environ.get(name, "")
    if not v:
        return default
    # ASCII string override; encode utf-8 bytes
    return v.encode("utf-8")

DOMAIN_TX = _domain_env("MA_DOMAIN_TX", b"MA_TX")
DOMAIN_MERKLE_LEAF = _domain_env("MA_DOMAIN_MERKLE_LEAF", b"MA_LEAF_V1")
DOMAIN_MERKLE_NODE = _domain_env("MA_DOMAIN_MERKLE_NODE", b"MA_NODE_V1")
DOMAIN_TXID = _domain_env("MA_DOMAIN_TXID", b"TXID_V1")
DOMAIN_SIGPACK = _domain_env("MA_DOMAIN_SIGPACK", b"SIGPACK_V1")
DOMAIN_RCPT_HASH = _domain_env("MA_DOMAIN_RCPT_HASH", b"RCPT_HASH_V1")

# ----------------------------- Utilities ---------------------------------------------

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def sha512(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()

def u32(x: int) -> int:
    return x & 0xFFFFFFFF

def u64(x: int) -> int:
    return x & 0xFFFFFFFFFFFFFFFF

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp_i32(x: int, lo: int, hi: int) -> int:
    if x < lo: return lo
    if x > hi: return hi
    return x

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def hexs(b: bytes) -> str:
    return b.hex()

def unhex(s: str) -> bytes:
    return bytes.fromhex(s)

def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def write_file(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    write_json(tmp, obj)
    os.replace(tmp, path)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, SNAP_DIR), exist_ok=True)

def backup_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    ts = int(time.time())
    bak = f"{path}.bak.{ts}"
    shutil.copy2(path, bak)
    return bak

def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

# ----------------------------- Ed25519 (RFC8032) ------------------------------------
# Pure-python reference implementation. Deterministic; not constant-time.

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, P-2, P) % P
I = pow(2, (P-1)//4, P)

def inv(x: int) -> int:
    return pow(x, P-2, P)

def edwards_add(P1: Tuple[int, int], P2: Tuple[int, int]) -> Tuple[int, int]:
    (x1, y1) = P1
    (x2, y2) = P2
    x3 = (x1 * y2 + x2 * y1) * inv(1 + D * x1 * x2 * y1 * y2) % P
    y3 = (y1 * y2 + x1 * x2) * inv(1 - D * x1 * x2 * y1 * y2) % P
    return (x3, y3)

def edwards_double(P1: Tuple[int, int]) -> Tuple[int, int]:
    return edwards_add(P1, P1)

def xrecover(y: int) -> int:
    xx = (y * y - 1) * inv(D * y * y + 1) % P
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P != 0:
        x = (x * I) % P
    if x % 2 != 0:
        x = P - x
    return x

By = 4 * inv(5) % P
Bx = xrecover(By)
B = (Bx, By)

def encodepoint(Pt: Tuple[int, int]) -> bytes:
    (x, y) = Pt
    y_int = y | ((x & 1) << 255)
    return y_int.to_bytes(32, "little")

def decodepoint(s: bytes) -> Tuple[int, int]:
    if len(s) != 32:
        raise ValueError("bad point length")
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    x_sign = (int.from_bytes(s, "little") >> 255) & 1
    x = xrecover(y)
    if (x & 1) != x_sign:
        x = P - x
    if (-x * x + y * y - 1 - D * x * x * y * y) % P != 0:
        raise ValueError("point not on curve")
    return (x, y)

def scalarmult(Pt: Tuple[int, int], e: int) -> Tuple[int, int]:
    Q = (0, 1)
    Pn = Pt
    while e > 0:
        if e & 1:
            Q = edwards_add(Q, Pn)
        Pn = edwards_double(Pn)
        e >>= 1
    return Q

def Hint(m: bytes) -> int:
    return int.from_bytes(sha512(m), "little")

def ed25519_keypair(seed32: bytes) -> Tuple[bytes, bytes]:
    if len(seed32) != 32:
        raise ValueError("seed must be 32 bytes")
    h = sha512(seed32)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    pk = encodepoint(scalarmult(B, a))
    return seed32, pk

def ed25519_sign(sk_seed32: bytes, msg: bytes) -> bytes:
    h = sha512(sk_seed32)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    prefix = h[32:]
    A = encodepoint(scalarmult(B, a))
    r = Hint(prefix + msg) % L
    R = encodepoint(scalarmult(B, r))
    k = Hint(R + A + msg) % L
    S = (r + k * a) % L
    return R + S.to_bytes(32, "little")

def ed25519_verify(pk32: bytes, msg: bytes, sig64: bytes) -> bool:
    try:
        if len(pk32) != 32 or len(sig64) != 64:
            return False
        Rb = sig64[:32]
        S = int.from_bytes(sig64[32:], "little")
        if S >= L:
            return False
        A = decodepoint(pk32)
        R = decodepoint(Rb)
        k = Hint(Rb + pk32 + msg) % L
        SB = scalarmult(B, S)
        RkA = edwards_add(R, scalarmult(A, k))
        return SB == RkA
    except Exception:
        return False

# ----------------------------- Merkle (MACv1) ---------------------------------------

def merkle_leaf(i: int, v_q16_16: int) -> bytes:
    payload = struct.pack("<H i", i, int(v_q16_16))
    return sha256(DOMAIN_MERKLE_LEAF + payload)

def merkle_root_macv1(state_q16_16: List[int]) -> bytes:
    if len(state_q16_16) != DIM:
        raise ValueError("bad dim for merkle root")
    lvl = [merkle_leaf(i, state_q16_16[i]) for i in range(DIM)]
    while len(lvl) > 1:
        nxt = []
        for j in range(0, len(lvl), 2):
            nxt.append(sha256(DOMAIN_MERKLE_NODE + lvl[j] + lvl[j + 1]))
        lvl = nxt
    return lvl[0]

def merkle_proof_macv1(state_q16_16: List[int], index: int) -> List[bytes]:
    if not (0 <= index < DIM):
        raise ValueError("index out of range")
    lvl = [merkle_leaf(i, state_q16_16[i]) for i in range(DIM)]
    proof: List[bytes] = []
    idx = index
    while len(lvl) > 1:
        sib = idx ^ 1
        proof.append(lvl[sib])
        nxt = []
        for j in range(0, len(lvl), 2):
            nxt.append(sha256(DOMAIN_MERKLE_NODE + lvl[j] + lvl[j + 1]))
        lvl = nxt
        idx //= 2
    return proof

# ----------------------------- Policy ------------------------------------------------

@dataclasses.dataclass
class EvidenceRule:
    evidence_type: int
    min_signers: int
    required_signers: List[str]  # pubkey hex; empty => use policy allowlist

    def normalized_required(self) -> set[str]:
        out = set()
        for h in self.required_signers:
            hh = str(h).lower().strip()
            if re.fullmatch(r"[0-9a-f]{64}", hh):
                out.add(hh)
        return out

@dataclasses.dataclass
class Policy:
    schema: str
    chain_id: str
    threshold: int
    allowlist: List[str]  # pubkey hex
    evidence_rules: List[EvidenceRule]

    def normalized_allowset(self) -> set[str]:
        out = set()
        for h in self.allowlist:
            hh = str(h).lower().strip()
            if re.fullmatch(r"[0-9a-f]{64}", hh):
                out.add(hh)
        return out

def default_policy(chain_id: int) -> Policy:
    return Policy(schema="MA_POLICY_V1.6", chain_id=str(int(chain_id)), threshold=1, allowlist=[], evidence_rules=[])

def _parse_evidence_rules(obj: Any) -> List[EvidenceRule]:
    rules: List[EvidenceRule] = []
    if not obj:
        return rules
    if not isinstance(obj, list):
        return rules
    for it in obj:
        if not isinstance(it, dict):
            continue
        try:
            et = int(it.get("evidence_type", -1))
            ms = int(it.get("min_signers", 1))
            rs = it.get("required_signers", [])
            if not isinstance(rs, list):
                rs = []
            ms = max(1, ms)
            if et < 0 or et > 255:
                continue
            rules.append(EvidenceRule(evidence_type=et, min_signers=ms, required_signers=[str(x) for x in rs]))
        except Exception:
            continue
    # stable ordering by type
    rules.sort(key=lambda r: r.evidence_type)
    return rules

def load_policy(data_dir: str, chain_id: int) -> Policy:
    path = os.path.join(data_dir, POLICY_JSON)
    if not os.path.exists(path):
        pol = default_policy(chain_id)
        atomic_write_json(path, {
            "schema": pol.schema,
            "chain_id": pol.chain_id,
            "threshold": pol.threshold,
            "allowlist": pol.allowlist,
            "evidence_rules": [],
        })
        return pol
    try:
        obj = read_json(path)
        if not isinstance(obj, dict):
            raise ValueError("bad policy.json")
        schema = str(obj.get("schema", ""))
        if schema not in ("MA_POLICY_V1.5", "MA_POLICY_V1.6"):
            raise ValueError("bad policy schema")
        if str(obj.get("chain_id", "")) != str(int(chain_id)):
            raise ValueError("policy chain_id mismatch")
        thr = int(obj.get("threshold", 1))
        allow = obj.get("allowlist", [])
        if not isinstance(allow, list):
            raise ValueError("policy allowlist must be list")

        # v1.5 had no evidence_rules; treat missing as empty
        rules = _parse_evidence_rules(obj.get("evidence_rules", []))

        pol = Policy(
            schema="MA_POLICY_V1.6",
            chain_id=str(int(chain_id)),
            threshold=max(1, thr),
            allowlist=[str(x) for x in allow],
            evidence_rules=rules,
        )
        # upgrade file on load if old schema
        if schema != "MA_POLICY_V1.6":
            atomic_write_json(path, {
                "schema": pol.schema,
                "chain_id": pol.chain_id,
                "threshold": pol.threshold,
                "allowlist": pol.allowlist,
                "evidence_rules": [dataclasses.asdict(r) for r in pol.evidence_rules],
            })
        return pol
    except Exception:
        return default_policy(chain_id)

def save_policy(data_dir: str, pol: Policy) -> None:
    path = os.path.join(data_dir, POLICY_JSON)
    atomic_write_json(path, {
        "schema": "MA_POLICY_V1.6",
        "chain_id": pol.chain_id,
        "threshold": int(pol.threshold),
        "allowlist": pol.allowlist,
        "evidence_rules": [dataclasses.asdict(r) for r in pol.evidence_rules],
    })

def policy_find_rule(pol: Policy, evidence_type: int) -> Optional[EvidenceRule]:
    for r in pol.evidence_rules:
        if int(r.evidence_type) == int(evidence_type):
            return r
    return None

# ----------------------------- TX / Receipt -----------------------------------------

@dataclasses.dataclass(frozen=True)
class MaDeltaHeaderV1:
    magic: bytes
    version: int
    dim: int
    op: int
    flags: int
    reserved0: int
    alpha_q0_32: int
    eps_q32_32: int
    eta_max_q0_32: int
    nonce: int
    timestamp_ms: int
    chain_id: int
    payload_len: int
    reserved1: bytes

    def pack(self) -> bytes:
        if self.magic != MAGIC: raise ValueError("bad magic")
        if self.version != VERSION: raise ValueError("bad version")
        if self.dim != DIM: raise ValueError("bad dim")
        if self.payload_len < self.dim * 2: raise ValueError("payload_len must be >= dim*2")
        if self.op != OP_PERFECT_T: raise ValueError("unsupported op")
        if self.reserved0 != 0: raise ValueError("reserved0 must be 0")
        if self.reserved1 != b"\x00" * 12: raise ValueError("reserved1 must be 12 zero bytes")
        return struct.pack(
            HEADER_FMT,
            self.magic,
            self.version,
            self.dim,
            self.op,
            self.flags,
            self.reserved0,
            u32(self.alpha_q0_32),
            u64(self.eps_q32_32),
            u32(self.eta_max_q0_32),
            u32(self.nonce),
            u64(self.timestamp_ms),
            u64(self.chain_id),
            u32(self.payload_len),
            self.reserved1,
        )

    @staticmethod
    def unpack(b: bytes) -> "MaDeltaHeaderV1":
        if len(b) < HEADER_SIZE: raise ValueError("short header")
        tup = struct.unpack(HEADER_FMT, b[:HEADER_SIZE])
        h = MaDeltaHeaderV1(*tup)
        if h.magic != MAGIC: raise ValueError("bad magic")
        if h.version != VERSION: raise ValueError("bad version")
        if h.dim != DIM: raise ValueError("bad dim")
        if h.payload_len < h.dim * 2: raise ValueError("bad payload_len (must be >= dim*2)")
        if h.op != OP_PERFECT_T: raise ValueError("unsupported op")
        if h.reserved0 != 0: raise ValueError("reserved0 must be 0")
        if h.reserved1 != b"\x00" * 12: raise ValueError("reserved1 must be 12 zero bytes")
        known = (FLAG_CLIP | FLAG_ETA_CLAMP | FLAG_INCLUDE_PUBKEY | FLAG_MULTI_SIG | FLAG_HAS_TLV)
        if h.flags & ~known:
            raise ValueError("unknown flags set")
        return h

def tx_message(chain_id: int, header: bytes, payload: bytes) -> bytes:
    return sha512(DOMAIN_TX + struct.pack("<Q", u64(chain_id)) + header + payload)

@dataclasses.dataclass
class SigPair:
    pubkey: bytes
    sig: bytes

@dataclasses.dataclass
class ParsedTx:
    header: MaDeltaHeaderV1
    header_bytes: bytes
    payload: bytes
    targets_payload: bytes
    tlv: bytes
    sigpairs: List[SigPair]
    raw: bytes

def parse_tx_bytes(tx: bytes) -> ParsedTx:
    if len(tx) < HEADER_SIZE:
        raise ValueError("tx too short")
    h = MaDeltaHeaderV1.unpack(tx[:HEADER_SIZE])
    header_bytes = tx[:HEADER_SIZE]

    payload_start = HEADER_SIZE
    payload_end = payload_start + int(h.payload_len)
    if len(tx) < payload_end:
        raise ValueError("short payload")
    payload = tx[payload_start:payload_end]
    targets_payload = payload[:DIM * 2]

    if (h.flags & FLAG_HAS_TLV) != 0:
        tlv = payload[DIM * 2:]
    else:
        if len(payload) > DIM * 2:
            raise ValueError("payload_len > 512 but FLAG_HAS_TLV not set")
        tlv = b""

    sigpairs: List[SigPair] = []
    idx = payload_end

    if (h.flags & FLAG_INCLUDE_PUBKEY) != 0:
        if len(tx) < idx + 32 + 64:
            raise ValueError("signed tx missing pk/sig")
        pk = tx[idx:idx+32]; sg = tx[idx+32:idx+32+64]
        sigpairs.append(SigPair(pk, sg))
        idx += 32 + 64

        if (h.flags & FLAG_MULTI_SIG) != 0:
            if len(tx) < idx + 1:
                raise ValueError("multisig missing count")
            extra_n = tx[idx]
            idx += 1
            for _ in range(int(extra_n)):
                if len(tx) < idx + 32 + 64:
                    raise ValueError("multisig truncated")
                pk2 = tx[idx:idx+32]; sg2 = tx[idx+32:idx+32+64]
                sigpairs.append(SigPair(pk2, sg2))
                idx += 32 + 64

    if idx != len(tx):
        raise ValueError("unexpected trailing bytes")

    return ParsedTx(h, header_bytes, payload, targets_payload, tlv, sigpairs, tx)

def payload_to_targets_q16_16(targets_payload_512: bytes) -> List[int]:
    if len(targets_payload_512) != DIM * 2:
        raise ValueError("targets payload len must be 512")
    out: List[int] = []
    for i in range(DIM):
        t_q8_8 = struct.unpack_from("<h", targets_payload_512, i * 2)[0]
        out.append(int(t_q8_8) << 8)  # Q8.8 -> Q16.16
    return out

@dataclasses.dataclass
class Evidence:
    present: bool
    evidence_type: int
    evidence_hash: bytes  # 32

def parse_tlv_evidence(tlv: bytes) -> Evidence:
    if not tlv:
        return Evidence(False, 0, b"")
    i = 0
    found = None
    while i + 3 <= len(tlv):
        t = tlv[i]
        ln = int.from_bytes(tlv[i+1:i+3], "little")
        i += 3
        if i + ln > len(tlv):
            break
        val = tlv[i:i+ln]
        i += ln
        if t == TLV_EVIDENCE:
            if ln != 1 + 32:
                continue
            et = val[0]
            eh = val[1:33]
            found = Evidence(True, int(et), eh)
    if found is None:
        return Evidence(False, 0, b"")
    return found

def compute_txid(header: bytes, payload: bytes, sigpairs: List[SigPair]) -> bytes:
    if sigpairs:
        chunks = sorted([sp.pubkey + sp.sig for sp in sigpairs])
        sigpack_hash = sha256(DOMAIN_SIGPACK + b"".join(chunks))
    else:
        sigpack_hash = b"\x00" * 32
    return sha256(DOMAIN_TXID + header + payload + sigpack_hash)

def receipt_hash(core: bytes) -> bytes:
    return sha256(DOMAIN_RCPT_HASH + core)

# ----------------------------- Apply (Perfect-T) ------------------------------------

@dataclasses.dataclass
class ApplyResult:
    pre_root: bytes
    post_root: bytes
    pre_d2_q32_32: int
    post_d2_q32_32: int
    denom_q32_32: int
    eta_q0_32: int
    clipped: bool

def calc_d2_q32_32(state_q16_16: List[int], targets_q16_16: List[int]) -> int:
    d2 = 0
    for i in range(DIM):
        e = int(state_q16_16[i]) - int(targets_q16_16[i])
        d2 = u64(d2 + (e * e))
    return int(d2)

def apply_perfect_t(
    state_q16_16: List[int],
    targets_q16_16: List[int],
    alpha_q0_32: int,
    eps_q32_32: int,
    eta_max_q0_32: int,
    do_clip: bool,
    clip_min_q16_16: int = DEFAULT_CLIP_MIN_Q16_16,
    clip_max_q16_16: int = DEFAULT_CLIP_MAX_Q16_16,
) -> ApplyResult:
    pre_root = merkle_root_macv1(state_q16_16)
    pre_d2 = calc_d2_q32_32(state_q16_16, targets_q16_16)

    e = [int(state_q16_16[i]) - int(targets_q16_16[i]) for i in range(DIM)]
    d2 = 0
    for i in range(DIM):
        ei = int(e[i])
        d2 = u64(d2 + (ei * ei))
    denom = u64(d2 + u64(eps_q32_32))

    if denom == 0:
        eta = 0
    else:
        eta = u64((u64(alpha_q0_32) << 32) // denom)

    if eta_max_q0_32 != 0:
        eta = min(eta, u64(eta_max_q0_32))

    clipped = False
    for i in range(DIM):
        delta = int((int(eta) * int(e[i])) >> 32)  # Q16.16
        v = int(state_q16_16[i]) - delta
        if do_clip:
            vv = clamp_i32(v, clip_min_q16_16, clip_max_q16_16)
            if vv != v:
                clipped = True
            v = vv
        state_q16_16[i] = int(v)

    post_root = merkle_root_macv1(state_q16_16)
    post_d2 = calc_d2_q32_32(state_q16_16, targets_q16_16)

    return ApplyResult(pre_root, post_root, int(pre_d2), int(post_d2), int(denom), int(eta), clipped)

# ----------------------------- Validation / Gates ------------------------------------

class TxReject(Exception):
    pass

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise TxReject(msg)

@dataclasses.dataclass
class NodeConfig:
    chain_id: int
    snapshot_every: int
    keep_snapshots: int
    require_signed: bool
    eta_cap_q0_32: int
    max_submit_bytes: int
    safe_push_full: bool
    policy: Policy

def default_config(chain_id: int, data_dir: str) -> NodeConfig:
    pol = load_policy(data_dir, chain_id)
    return NodeConfig(
        chain_id=chain_id,
        snapshot_every=DEFAULT_SNAPSHOT_EVERY,
        keep_snapshots=DEFAULT_KEEP_SNAPSHOTS,
        require_signed=True,
        eta_cap_q0_32=DEFAULT_ETA_CAP_Q0_32,
        max_submit_bytes=DEFAULT_MAX_SUBMIT_BYTES,
        safe_push_full=True,
        policy=pol,
    )

def validate_tx_struct(ptx: ParsedTx, cfg: NodeConfig) -> None:
    h = ptx.header
    require(h.chain_id == cfg.chain_id, "wrong chain_id")
    require(h.dim == DIM, "wrong dim")
    require(h.op == OP_PERFECT_T, "unsupported op")
    require(h.alpha_q0_32 >= 0, "alpha must be >=0")
    require(h.eps_q32_32 >= 0, "eps must be >=0")
    require(len(ptx.targets_payload) == DIM * 2, "bad targets payload")

    if cfg.require_signed:
        require((h.flags & FLAG_INCLUDE_PUBKEY) != 0, "signed tx required by node")
        require(len(ptx.sigpairs) >= 1, "missing signatures")
    else:
        if (h.flags & FLAG_INCLUDE_PUBKEY) != 0:
            require(len(ptx.sigpairs) >= 1, "flag says signed but missing bytes")

    if (h.flags & FLAG_MULTI_SIG) != 0:
        require((h.flags & FLAG_INCLUDE_PUBKEY) != 0, "MULTI_SIG requires INCLUDE_PUBKEY")

def verify_sigs_and_policy(ptx: ParsedTx, cfg: NodeConfig) -> List[str]:
    """
    Returns list of pubkey hex that are valid signers (unique), subject to allowlist + threshold.
    """
    h = ptx.header
    if (h.flags & FLAG_INCLUDE_PUBKEY) == 0:
        return []
    msg = tx_message(h.chain_id, ptx.header_bytes, ptx.payload)

    allow = cfg.policy.normalized_allowset()
    valid: List[str] = []
    seen: set[str] = set()

    for sp in ptx.sigpairs:
        pkhex = hexs(sp.pubkey)
        if pkhex in seen:
            continue
        if allow and (pkhex not in allow):
            continue
        if ed25519_verify(sp.pubkey, msg, sp.sig):
            seen.add(pkhex)
            valid.append(pkhex)

    if cfg.require_signed:
        thr = max(1, int(cfg.policy.threshold))
        require(len(valid) >= thr, f"policy threshold not met: have {len(valid)} need {thr}")
    return valid

def validate_evidence_rules(evidence: Evidence, valid_signers: List[str], pol: Policy) -> None:
    """
    If evidence present and a rule exists for evidence_type, enforce it.
    Rules are OPTIONAL; if no rule exists for evidence_type, evidence is just committed (not validated).
    """
    if not evidence.present:
        return
    rule = policy_find_rule(pol, int(evidence.evidence_type))
    if rule is None:
        return  # committed only, no validation constraints
    # evidence is present (by definition) — enforce signer requirements
    rs = rule.normalized_required()
    if not rs:
        rs = pol.normalized_allowset()  # empty required_signers => allowlist
    require(len(rs) > 0, "evidence rule has no signer set (allowlist empty)")
    have = 0
    vset = set([s.lower() for s in valid_signers])
    for s in rs:
        if s in vset:
            have += 1
    require(have >= max(1, int(rule.min_signers)), f"evidence rule not met: have {have} need {rule.min_signers}")

def gate_regression(apply: ApplyResult, h: MaDeltaHeaderV1, cfg: NodeConfig) -> None:
    require(0 <= apply.eta_q0_32 <= cfg.eta_cap_q0_32, "eta exceeds node cap")
    if apply.pre_d2_q32_32 != 0:
        require(apply.post_d2_q32_32 <= apply.pre_d2_q32_32, "distance did not decrease")
    if h.flags & FLAG_ETA_CLAMP:
        require(h.eta_max_q0_32 != 0, "FLAG_ETA_CLAMP set but eta_max_q0_32 == 0")
        require(apply.eta_q0_32 <= int(h.eta_max_q0_32), "eta exceeded tx eta_max")

# ----------------------------- Receipts / TXLOG / Snapshot ---------------------------

def make_receipt_core(
    chain_id: int,
    height: int,
    prev_receipt_hash: bytes,
    txid_bytes: bytes,
    pre_root: bytes,
    post_root: bytes,
    pre_d2_q32_32: int,
    post_d2_q32_32: int,
    eta_q0_32: int,
    nonce: int,
    timestamp_ms: int,
    evidence_type: int,
    evidence_hash32: bytes,
) -> bytes:
    evh = evidence_hash32 if evidence_hash32 else (b"\x00" * 32)
    return (
        struct.pack("<Q Q", u64(chain_id), u64(height)) +
        prev_receipt_hash +
        txid_bytes +
        pre_root +
        post_root +
        struct.pack("<Q Q I I Q", u64(pre_d2_q32_32), u64(post_d2_q32_32), u32(eta_q0_32), u32(nonce), u64(timestamp_ms)) +
        struct.pack("<B", u32(evidence_type) & 0xFF) +
        evh
    )

def receipt_json_v16(
    chain_id: int,
    height: int,
    prev_receipt_hash_hex: str,
    rcpt_hash_hex: str,
    txid_hex: str,
    header: MaDeltaHeaderV1,
    valid_signers: List[str],
    apply: ApplyResult,
    evidence: Evidence,
) -> Dict[str, Any]:
    j = {
        "schema": "MA_RECEIPT_V1.6",
        "chain_id": str(int(chain_id)),
        "height": int(height),
        "prev_receipt_hash": prev_receipt_hash_hex,
        "receipt_hash": rcpt_hash_hex,
        "txid": txid_hex,
        "op": int(header.op),
        "flags": int(header.flags),
        "nonce": int(header.nonce),
        "timestamp_ms": int(header.timestamp_ms),
        "alpha_q0_32": str(int(header.alpha_q0_32)),
        "eps_q32_32": str(int(header.eps_q32_32)),
        "eta_max_q0_32": str(int(header.eta_max_q0_32)),
        "pre_root": hexs(apply.pre_root),
        "post_root": hexs(apply.post_root),
        "pre_d2_q32_32": str(int(apply.pre_d2_q32_32)),
        "post_d2_q32_32": str(int(apply.post_d2_q32_32)),
        "denom_q32_32": str(int(apply.denom_q32_32)),
        "eta_q0_32": str(int(apply.eta_q0_32)),
        "clipped": bool(apply.clipped),
        "valid_signers": valid_signers,
        "domains": {  # helps audits prove same domain config
            "TX": DOMAIN_TX.decode("utf-8", "replace"),
            "LEAF": DOMAIN_MERKLE_LEAF.decode("utf-8", "replace"),
            "NODE": DOMAIN_MERKLE_NODE.decode("utf-8", "replace"),
            "TXID": DOMAIN_TXID.decode("utf-8", "replace"),
            "SIGPACK": DOMAIN_SIGPACK.decode("utf-8", "replace"),
            "RCPT": DOMAIN_RCPT_HASH.decode("utf-8", "replace"),
        },
    }
    if evidence.present:
        j["evidence_type"] = int(evidence.evidence_type)
        j["evidence_hash"] = hexs(evidence.evidence_hash)
    return j

def txlog_entry(height: int, txid_hex: str, tx_bytes: bytes) -> Dict[str, Any]:
    return {
        "schema": "MA_TXLOG_V1.6",
        "height": int(height),
        "txid": txid_hex,
        "tx_b64": b64e(tx_bytes),
    }

# ----------------------------- NodeDB ------------------------------------------------

class NodeDB:
    """
    data_dir/
      state.json
      policy.json
      receipts.jsonl
      txlog.jsonl
      snaps/snap_{height}_{root}.json
    """
    def __init__(self, data_dir: str, cfg: NodeConfig):
        self.data_dir = data_dir
        self.cfg = cfg
        self.state_path = os.path.join(data_dir, STATE_JSON)
        self.policy_path = os.path.join(data_dir, POLICY_JSON)
        self.receipts_path = os.path.join(data_dir, RECEIPTS_JSONL)
        self.txlog_path = os.path.join(data_dir, TXLOG_JSONL)
        self.snap_dir = os.path.join(data_dir, SNAP_DIR)
        ensure_dir(data_dir)

        self._lock = threading.RLock()
        self.state_q16_16: List[int] = [0] * DIM
        self.root: bytes = merkle_root_macv1(self.state_q16_16)
        self.height: int = 0
        self.head_receipt_hash: bytes = b"\x00" * 32

        self._rcpt_hash_by_h: Optional[Dict[int, bytes]] = None
        self._tx_by_h: Optional[Dict[int, Dict[str, Any]]] = None
        self._rcpt_by_h: Optional[Dict[int, Dict[str, Any]]] = None

        self._load_state_only()

    def _load_state_only(self) -> None:
        with self._lock:
            self.cfg.policy = load_policy(self.data_dir, self.cfg.chain_id)

            if os.path.exists(self.state_path):
                obj = read_json(self.state_path)
                arr = obj["state_q16_16"] if isinstance(obj, dict) and "state_q16_16" in obj else obj
                if not isinstance(arr, list) or len(arr) != DIM:
                    raise ValueError("bad state.json")
                self.state_q16_16 = [int(x) for x in arr]
                self.root = unhex(obj.get("root", hexs(merkle_root_macv1(self.state_q16_16))))
                self.height = int(obj.get("height", 0))
                self.head_receipt_hash = unhex(obj.get("head_receipt_hash", "00"*32))
            else:
                self.state_q16_16 = [0] * DIM
                self.root = merkle_root_macv1(self.state_q16_16)
                self.height = 0
                self.head_receipt_hash = b"\x00" * 32

            if not os.path.exists(self.receipts_path):
                write_file(self.receipts_path, b"")
            if not os.path.exists(self.txlog_path):
                write_file(self.txlog_path, b"")

    def invalidate_caches(self) -> None:
        self._rcpt_hash_by_h = None
        self._tx_by_h = None
        self._rcpt_by_h = None

    def save_state(self) -> None:
        with self._lock:
            self.root = merkle_root_macv1(self.state_q16_16)
            obj = {
                "schema": "MA_STATE_V1.6",
                "chain_id": str(int(self.cfg.chain_id)),
                "dim": DIM,
                "q": "Q16.16",
                "state_q16_16": self.state_q16_16,
                "root": hexs(self.root),
                "height": int(self.height),
                "head_receipt_hash": hexs(self.head_receipt_hash),
                "updated_ms": now_ms(),
            }
            atomic_write_json(self.state_path, obj)

    def append_receipt(self, r: Dict[str, Any]) -> None:
        with self._lock:
            append_jsonl(self.receipts_path, r)
            self.height = int(r["height"])
            self.head_receipt_hash = unhex(r["receipt_hash"])
            self.invalidate_caches()

    def append_txlog(self, t: Dict[str, Any]) -> None:
        with self._lock:
            append_jsonl(self.txlog_path, t)
            self.invalidate_caches()

    def latest_snapshot(self) -> Optional[str]:
        best_h = -1
        best_path = None
        for fn in os.listdir(self.snap_dir):
            m = re.match(r"^snap_(\d+)_([0-9a-f]{64})\.json$", fn)
            if not m: continue
            h = int(m.group(1))
            if h > best_h:
                best_h = h
                best_path = os.path.join(self.snap_dir, fn)
        return best_path

    def best_snapshot_at_or_below(self, height: int) -> Optional[str]:
        best_h = -1
        best_path = None
        for fn in os.listdir(self.snap_dir):
            m = re.match(r"^snap_(\d+)_([0-9a-f]{64})\.json$", fn)
            if not m: continue
            h = int(m.group(1))
            if h <= height and h > best_h:
                best_h = h
                best_path = os.path.join(self.snap_dir, fn)
        return best_path

    def write_snapshot(self) -> str:
        with self._lock:
            roothex = hexs(merkle_root_macv1(self.state_q16_16))
            fn = f"snap_{self.height}_{roothex}.json"
            path = os.path.join(self.snap_dir, fn)
            obj = {
                "schema": "MA_SNAPSHOT_V1.6",
                "chain_id": str(int(self.cfg.chain_id)),
                "height": int(self.height),
                "root": roothex,
                "state_q16_16": self.state_q16_16,
                "head_receipt_hash": hexs(self.head_receipt_hash),
                "created_ms": now_ms(),
            }
            atomic_write_json(path, obj)
            self._prune_snapshots_locked()
            return path

    def _prune_snapshots_locked(self) -> None:
        snaps: List[Tuple[int, str]] = []
        for fn in os.listdir(self.snap_dir):
            m = re.match(r"^snap_(\d+)_([0-9a-f]{64})\.json$", fn)
            if not m: continue
            snaps.append((int(m.group(1)), fn))
        snaps.sort(reverse=True)
        for _, fn in snaps[self.cfg.keep_snapshots:]:
            try: os.remove(os.path.join(self.snap_dir, fn))
            except OSError: pass

    def _build_receipt_index_locked(self) -> None:
        if self._rcpt_hash_by_h is not None and self._rcpt_by_h is not None:
            return
        rh: Dict[int, bytes] = {0: b"\x00"*32}
        rr: Dict[int, Dict[str, Any]] = {}
        for r in iter_jsonl(self.receipts_path):
            if not isinstance(r, dict): continue
            if "height" not in r or "receipt_hash" not in r:
                continue
            try:
                h = int(r.get("height", -1))
                if h <= 0: continue
                rr[h] = r
                rh[h] = unhex(r["receipt_hash"])
            except Exception:
                continue
        self._rcpt_hash_by_h = rh
        self._rcpt_by_h = rr

    def _build_tx_index_locked(self) -> None:
        if self._tx_by_h is not None:
            return
        tx: Dict[int, Dict[str, Any]] = {}
        for t in iter_jsonl(self.txlog_path):
            if not isinstance(t, dict): continue
            if t.get("schema") not in ("MA_TXLOG_V1.5", "MA_TXLOG_V1.6"):
                continue
            try:
                h = int(t.get("height", -1))
                if h <= 0: continue
                tx[h] = t
            except Exception:
                continue
        self._tx_by_h = tx

    def get_receipt_hash(self, height: int) -> Optional[bytes]:
        with self._lock:
            self._build_receipt_index_locked()
            assert self._rcpt_hash_by_h is not None
            return self._rcpt_hash_by_h.get(height)

    def get_receipt_by_height(self, height: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._build_receipt_index_locked()
            assert self._rcpt_by_h is not None
            return self._rcpt_by_h.get(height)

    def get_tx_by_height(self, height: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._build_tx_index_locked()
            assert self._tx_by_h is not None
            return self._tx_by_h.get(height)

    def get_receipts_range(self, start: int, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        end = start + limit
        for h in range(start, end):
            r = self.get_receipt_by_height(h)
            if r is None:
                break
            out.append(r)
        return out

    def get_txs_range(self, start: int, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        end = start + limit
        for h in range(start, end):
            t = self.get_tx_by_height(h)
            if t is None:
                break
            out.append(t)
        return out

# ----------------------------- Apply + Persist ---------------------------------------

def apply_tx_to_db(db: NodeDB, tx_bytes: bytes) -> Dict[str, Any]:
    cfg = db.cfg
    require(len(tx_bytes) <= cfg.max_submit_bytes, "tx too large")

    ptx = parse_tx_bytes(tx_bytes)
    validate_tx_struct(ptx, cfg)
    valid_signers = verify_sigs_and_policy(ptx, cfg)

    evidence = parse_tlv_evidence(ptx.tlv)
    # v1.6: evidence validation rules (optional)
    validate_evidence_rules(evidence, valid_signers, cfg.policy)

    with db._lock:
        targets = payload_to_targets_q16_16(ptx.targets_payload)
        eta_max = int(ptx.header.eta_max_q0_32) if (ptx.header.flags & FLAG_ETA_CLAMP) else 0

        apply = apply_perfect_t(
            state_q16_16=db.state_q16_16,
            targets_q16_16=targets,
            alpha_q0_32=int(ptx.header.alpha_q0_32),
            eps_q32_32=int(ptx.header.eps_q32_32),
            eta_max_q0_32=eta_max,
            do_clip=bool(ptx.header.flags & FLAG_CLIP),
        )
        gate_regression(apply, ptx.header, cfg)

        txid_bytes = compute_txid(ptx.header_bytes, ptx.payload, ptx.sigpairs)
        prev_hash = db.head_receipt_hash
        new_height = db.height + 1

        core = make_receipt_core(
            chain_id=cfg.chain_id,
            height=new_height,
            prev_receipt_hash=prev_hash,
            txid_bytes=txid_bytes,
            pre_root=apply.pre_root,
            post_root=apply.post_root,
            pre_d2_q32_32=apply.pre_d2_q32_32,
            post_d2_q32_32=apply.post_d2_q32_32,
            eta_q0_32=apply.eta_q0_32,
            nonce=ptx.header.nonce,
            timestamp_ms=ptx.header.timestamp_ms,
            evidence_type=int(evidence.evidence_type if evidence.present else 0),
            evidence_hash32=(evidence.evidence_hash if evidence.present else b""),
        )
        rcpt_hash = receipt_hash(core)

        rj = receipt_json_v16(
            chain_id=cfg.chain_id,
            height=new_height,
            prev_receipt_hash_hex=hexs(prev_hash),
            rcpt_hash_hex=hexs(rcpt_hash),
            txid_hex=hexs(txid_bytes),
            header=ptx.header,
            valid_signers=valid_signers,
            apply=apply,
            evidence=evidence,
        )

        db.append_txlog(txlog_entry(new_height, rj["txid"], tx_bytes))
        db.append_receipt(rj)
        db.save_state()

        snap_path = ""
        if cfg.snapshot_every > 0 and (db.height % cfg.snapshot_every == 0):
            snap_path = db.write_snapshot()

        return {
            "ok": True,
            "height": int(db.height),
            "root": hexs(db.root),
            "head_receipt_hash": hexs(db.head_receipt_hash),
            "txid": rj["txid"],
            "receipt_hash": rj["receipt_hash"],
            "pre_root": rj["pre_root"],
            "post_root": rj["post_root"],
            "eta_q0_32": rj["eta_q0_32"],
            "pre_d2_q32_32": rj["pre_d2_q32_32"],
            "post_d2_q32_32": rj["post_d2_q32_32"],
            "valid_signers": valid_signers,
            "evidence": ({"type": evidence.evidence_type, "hash": hexs(evidence.evidence_hash)} if evidence.present else None),
            "snapshot": snap_path,
        }

# ----------------------------- Snapshot-aware Replay ---------------------------------

@dataclasses.dataclass
class ReplayResult:
    ok: bool
    scanned: int
    verified: int
    computed_height: int
    computed_root: str
    computed_head_receipt_hash: str
    db_height: int
    db_root: str
    db_head_receipt_hash: str
    matches_db: bool
    snapshot_used: bool
    snapshot_height: int
    snapshot_path: str
    errors: List[str]

def load_snapshot(path: str) -> Tuple[int, bytes, bytes, List[int]]:
    obj = read_json(path)
    if not isinstance(obj, dict) or "height" not in obj or "root" not in obj or "state_q16_16" not in obj:
        raise ValueError("bad snapshot schema")
    h = int(obj["height"])
    state = [int(x) for x in obj["state_q16_16"]]
    if len(state) != DIM:
        raise ValueError("bad snapshot dim")
    root = unhex(str(obj["root"]))
    head_rcpt = unhex(str(obj.get("head_receipt_hash", "00"*32)))
    rr = merkle_root_macv1(state)
    if rr != root:
        raise ValueError("snapshot root mismatch to state")
    return h, root, head_rcpt, state

def build_maps_from_files(receipts_path: str, txlog_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    receipts_by_h: Dict[int, Dict[str, Any]] = {}
    for r in iter_jsonl(receipts_path):
        if not isinstance(r, dict): continue
        if "height" not in r or "receipt_hash" not in r:
            continue
        try:
            h = int(r.get("height", -1))
            if h > 0:
                receipts_by_h[h] = r
        except Exception:
            continue

    tx_by_h: Dict[int, Dict[str, Any]] = {}
    for t in iter_jsonl(txlog_path):
        if not isinstance(t, dict): continue
        if t.get("schema") not in ("MA_TXLOG_V1.5", "MA_TXLOG_V1.6"):
            continue
        try:
            h = int(t.get("height", -1))
            if h > 0:
                tx_by_h[h] = t
        except Exception:
            continue

    return receipts_by_h, tx_by_h

def full_replay_from_logs_with_optional_snapshot(
    receipts_by_h: Dict[int, Dict[str, Any]],
    tx_by_h: Dict[int, Dict[str, Any]],
    cfg: NodeConfig,
    snapshot_path: Optional[str],
    max_height: int,
) -> Tuple[bool, int, int, int, bytes, bytes, List[int], bool, int, str, List[str]]:
    errors: List[str] = []
    state = [0] * DIM
    computed_root = merkle_root_macv1(state)
    computed_rcpt = b"\x00" * 32
    start_h = 1
    snapshot_used = False
    snap_h = 0
    snap_p = snapshot_path or ""

    if snapshot_path:
        try:
            sh, sroot, srcpt, sstate = load_snapshot(snapshot_path)
            if 0 < sh <= max_height:
                r = receipts_by_h.get(sh)
                if r is not None:
                    if str(r.get("receipt_hash", "")) == hexs(srcpt) and str(r.get("post_root", "")) == hexs(sroot):
                        state = list(sstate)
                        computed_root = sroot
                        computed_rcpt = srcpt
                        start_h = sh + 1
                        snapshot_used = True
                        snap_h = sh
        except Exception:
            snapshot_used = False
            snap_h = 0

    scanned = 0
    verified = 0

    for h in range(start_h, max_height + 1):
        scanned += 1
        r = receipts_by_h.get(h)
        t = tx_by_h.get(h)
        if r is None:
            errors.append(f"missing receipt at height {h}")
            break
        if t is None:
            errors.append(f"missing txlog at height {h}")
            break

        try:
            if r.get("prev_receipt_hash", "") != hexs(computed_rcpt):
                raise TxReject(f"prev_receipt_hash mismatch at height {h}")

            tx_bytes = b64d(t["tx_b64"])
            ptx = parse_tx_bytes(tx_bytes)
            validate_tx_struct(ptx, cfg)
            valid_signers = verify_sigs_and_policy(ptx, cfg)

            evidence = parse_tlv_evidence(ptx.tlv)
            validate_evidence_rules(evidence, valid_signers, cfg.policy)

            txid_bytes = compute_txid(ptx.header_bytes, ptx.payload, ptx.sigpairs)
            if hexs(txid_bytes) != str(r.get("txid", "")):
                raise TxReject(f"txid mismatch at height {h}")

            if r.get("pre_root", "") != hexs(computed_root):
                raise TxReject(f"pre_root mismatch at height {h}")

            targets = payload_to_targets_q16_16(ptx.targets_payload)
            eta_max = int(ptx.header.eta_max_q0_32) if (ptx.header.flags & FLAG_ETA_CLAMP) else 0
            apply = apply_perfect_t(
                state_q16_16=state,
                targets_q16_16=targets,
                alpha_q0_32=int(ptx.header.alpha_q0_32),
                eps_q32_32=int(ptx.header.eps_q32_32),
                eta_max_q0_32=eta_max,
                do_clip=bool(ptx.header.flags & FLAG_CLIP),
            )
            gate_regression(apply, ptx.header, cfg)

            if r.get("post_root", "") != hexs(apply.post_root):
                raise TxReject(f"post_root mismatch at height {h}")
            if str(int(apply.pre_d2_q32_32)) != str(r.get("pre_d2_q32_32", "")):
                raise TxReject(f"pre_d2 mismatch at height {h}")
            if str(int(apply.post_d2_q32_32)) != str(r.get("post_d2_q32_32", "")):
                raise TxReject(f"post_d2 mismatch at height {h}")
            if str(int(apply.eta_q0_32)) != str(r.get("eta_q0_32", "")):
                raise TxReject(f"eta mismatch at height {h}")

            exp_ev_type = int(evidence.evidence_type if evidence.present else 0)
            exp_ev_hash = evidence.evidence_hash if evidence.present else (b"\x00"*32)

            got_ev_type = int(r.get("evidence_type", 0)) if "evidence_type" in r else 0
            got_ev_hash = unhex(r.get("evidence_hash", "00"*64)) if "evidence_hash" in r else (b"\x00"*32)

            if got_ev_type != exp_ev_type:
                raise TxReject(f"evidence_type mismatch at height {h}")
            if got_ev_hash != exp_ev_hash:
                raise TxReject(f"evidence_hash mismatch at height {h}")

            if "valid_signers" in r:
                rs = r.get("valid_signers", [])
                if isinstance(rs, list):
                    rs_set = set(str(x) for x in rs)
                    if not rs_set.issubset(set(valid_signers)):
                        raise TxReject(f"valid_signers mismatch at height {h}")

            core = make_receipt_core(
                chain_id=cfg.chain_id,
                height=h,
                prev_receipt_hash=computed_rcpt,
                txid_bytes=txid_bytes,
                pre_root=unhex(r["pre_root"]),
                post_root=unhex(r["post_root"]),
                pre_d2_q32_32=int(r["pre_d2_q32_32"]),
                post_d2_q32_32=int(r["post_d2_q32_32"]),
                eta_q0_32=int(r["eta_q0_32"]),
                nonce=int(r["nonce"]),
                timestamp_ms=int(r["timestamp_ms"]),
                evidence_type=exp_ev_type,
                evidence_hash32=(exp_ev_hash if evidence.present else b""),
            )
            rh = receipt_hash(core)
            if hexs(rh) != r.get("receipt_hash", ""):
                raise TxReject(f"receipt_hash mismatch at height {h}")

            computed_rcpt = rh
            computed_root = apply.post_root
            verified += 1

        except Exception as e:
            errors.append(str(e))
            break

    computed_height = max_height if len(errors) == 0 else (start_h - 1 + verified)
    return (len(errors) == 0), scanned, verified, computed_height, computed_root, computed_rcpt, state, snapshot_used, snap_h, snap_p, errors

def full_replay_db(db: NodeDB, commit: bool = False) -> ReplayResult:
    cfg = db.cfg
    with db._lock:
        receipts_by_h, tx_by_h = build_maps_from_files(db.receipts_path, db.txlog_path)
        max_h = max(receipts_by_h.keys(), default=0)

        snap_path = db.best_snapshot_at_or_below(max_h)
        ok, scanned, verified, ch, croot, crcpt, state, snap_used, snap_h, snap_p, errors = \
            full_replay_from_logs_with_optional_snapshot(receipts_by_h, tx_by_h, cfg, snap_path, max_h)

        matches_db = (ch == db.height and hexs(croot) == hexs(db.root) and hexs(crcpt) == hexs(db.head_receipt_hash))

        if commit and ok:
            db.state_q16_16 = state
            db.height = ch
            db.head_receipt_hash = crcpt
            db.save_state()
            if db.cfg.snapshot_every > 0 and (db.height % db.cfg.snapshot_every == 0):
                db.write_snapshot()

        return ReplayResult(
            ok=ok,
            scanned=int(scanned),
            verified=int(verified),
            computed_height=int(ch),
            computed_root=hexs(croot),
            computed_head_receipt_hash=hexs(crcpt),
            db_height=int(db.height),
            db_root=hexs(db.root),
            db_head_receipt_hash=hexs(db.head_receipt_hash),
            matches_db=bool(matches_db),
            snapshot_used=bool(snap_used),
            snapshot_height=int(snap_h),
            snapshot_path=str(snap_p),
            errors=errors,
        )

# ----------------------------- HTTP Client Helpers -----------------------------------

def _parse_base_url(url: str) -> Tuple[str, int]:
    m = re.match(r"^https?://([^/:]+)(?::(\d+))?$", url.strip("/"))
    if not m:
        raise ValueError("url must be like http://host:port")
    host = m.group(1)
    port = int(m.group(2) or "80")
    return host, port

def http_get_json(url: str, path: str, timeout_s: int = DEFAULT_HTTP_TIMEOUT_S) -> Tuple[int, Any]:
    host, port = _parse_base_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    try:
        j = json.loads(body.decode("utf-8"))
    except Exception:
        j = {"raw": body.decode("utf-8", "replace")}
    return resp.status, j

def http_post_json(url: str, path: str, obj: Any, timeout_s: int = DEFAULT_HTTP_TIMEOUT_S) -> Tuple[int, Any]:
    host, port = _parse_base_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    data = json.dumps(obj).encode("utf-8")
    conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    try:
        j = json.loads(body.decode("utf-8"))
    except Exception:
        j = {"raw": body.decode("utf-8", "replace")}
    return resp.status, j

# ----------------------------- Fork-choice + Reconcile --------------------------------

@dataclasses.dataclass
class PeerHead:
    peer: str
    ok: bool
    chain_id: int
    height: int
    head_receipt_hash: str
    root: str
    error: str

@dataclasses.dataclass
class ReconcileAttempt:
    peer: str
    ok: bool
    adopted: bool
    reason: str
    ancestor_height: int
    peer_height: int
    local_height_before: int
    local_height_after: int
    snapshot_used: bool
    snapshot_height: int
    errors: List[str]

def fork_choice_better(height_a: int, head_a_hex: str, height_b: int, head_b_hex: str) -> bool:
    if height_a != height_b:
        return height_a > height_b
    return head_a_hex > head_b_hex

def fetch_peer_head(peer: str) -> PeerHead:
    try:
        st, j = http_post_json(peer, "/sync/heads", {}, timeout_s=DEFAULT_HTTP_TIMEOUT_S)
        if st != 200 or not isinstance(j, dict) or not j.get("ok"):
            return PeerHead(peer, False, 0, 0, "", "", f"bad response status={st}")
        chain_id = int(j.get("chain_id", "0"))
        return PeerHead(peer, True, chain_id, int(j.get("height", 0)),
                        str(j.get("head_receipt_hash", "")), str(j.get("root", "")), "")
    except Exception as e:
        return PeerHead(peer, False, 0, 0, "", "", str(e))

def peer_receipt_hash(peer: str, height: int) -> Optional[str]:
    st, j = http_get_json(peer, f"/receipt/hash/{height}", timeout_s=DEFAULT_HTTP_TIMEOUT_S)
    if st != 200 or not isinstance(j, dict) or not j.get("ok"):
        return None
    return str(j.get("receipt_hash", ""))

def find_common_ancestor(db: NodeDB, peer: str, peer_height: int) -> int:
    lo = 0
    hi = min(db.height, peer_height)
    if hi == 0:
        return 0

    local_hi = db.get_receipt_hash(hi)
    if local_hi is not None:
        ph = peer_receipt_hash(peer, hi)
        if ph is not None and ph == hexs(local_hi):
            return hi

    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            best = 0
            lo = 1
            continue
        local_mid = db.get_receipt_hash(mid)
        if local_mid is None:
            hi = mid - 1
            continue
        ph = peer_receipt_hash(peer, mid)
        if ph is None:
            hi = mid - 1
            continue
        if ph == hexs(local_mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def pull_peer_full(peer: str, start_height: int, limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    st, j = http_post_json(peer, "/sync/pull_full", {"from_height": start_height, "limit": limit}, timeout_s=DEFAULT_HTTP_TIMEOUT_S)
    if st != 200 or not isinstance(j, dict) or not j.get("ok"):
        raise TxReject(f"peer pull_full failed status={st}")
    recs = j.get("receipts", [])
    txs = j.get("txs", [])
    if not isinstance(recs, list) or not isinstance(txs, list):
        raise TxReject("peer pull_full returned non-lists")
    return recs, txs

def build_local_maps_upto(db: NodeDB, upto_height: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    receipts_by_h: Dict[int, Dict[str, Any]] = {}
    tx_by_h: Dict[int, Dict[str, Any]] = {}
    for h in range(1, upto_height + 1):
        r = db.get_receipt_by_height(h)
        t = db.get_tx_by_height(h)
        if r is None or t is None:
            break
        receipts_by_h[h] = r
        tx_by_h[h] = t
    return receipts_by_h, tx_by_h

def merge_with_peer_suffix(
    local_receipts: Dict[int, Dict[str, Any]],
    local_txs: Dict[int, Dict[str, Any]],
    peer_recs: List[Dict[str, Any]],
    peer_txs: List[Dict[str, Any]],
    start_height: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    receipts_by_h = dict(local_receipts)
    tx_by_h = dict(local_txs)

    peer_tx_by_h: Dict[int, Dict[str, Any]] = {}
    for t in peer_txs:
        if isinstance(t, dict) and t.get("schema") in ("MA_TXLOG_V1.5", "MA_TXLOG_V1.6"):
            try:
                peer_tx_by_h[int(t.get("height", -1))] = t
            except Exception:
                pass

    expected = start_height
    for r in peer_recs:
        if not isinstance(r, dict):
            continue
        try:
            h = int(r.get("height", -1))
        except Exception:
            continue
        if h != expected:
            break
        t = peer_tx_by_h.get(h)
        if t is None:
            break
        receipts_by_h[h] = r
        tx_by_h[h] = t
        expected += 1

    return receipts_by_h, tx_by_h

def adopt_chain_atomic(db: NodeDB, receipts_by_h: Dict[int, Dict[str, Any]], tx_by_h: Dict[int, Dict[str, Any]],
                      final_state: List[int], final_height: int, final_root: bytes, final_head: bytes) -> None:
    with db._lock:
        backup_file(db.receipts_path)
        backup_file(db.txlog_path)
        backup_file(db.state_path)

        rec_tmp = db.receipts_path + ".tmp"
        tx_tmp = db.txlog_path + ".tmp"

        with open(rec_tmp, "w", encoding="utf-8") as f:
            for h in range(1, final_height + 1):
                r = receipts_by_h.get(h)
                if r is None:
                    raise TxReject(f"adopt missing receipt at height {h}")
                f.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")

        with open(tx_tmp, "w", encoding="utf-8") as f:
            for h in range(1, final_height + 1):
                t = tx_by_h.get(h)
                if t is None:
                    raise TxReject(f"adopt missing tx at height {h}")
                f.write(json.dumps(t, separators=(",", ":"), sort_keys=True) + "\n")

        os.replace(rec_tmp, db.receipts_path)
        os.replace(tx_tmp, db.txlog_path)

        db.state_q16_16 = final_state
        db.height = final_height
        db.root = final_root
        db.head_receipt_hash = final_head
        db.save_state()

        if db.cfg.snapshot_every > 0 and (db.height % db.cfg.snapshot_every == 0):
            db.write_snapshot()

        db.invalidate_caches()

def reconcile_with_peer(db: NodeDB, peer: str, pull_chunk: int = DEFAULT_MAX_PULL) -> ReconcileAttempt:
    errors: List[str] = []
    local_before = db.height

    ph = fetch_peer_head(peer)
    if not ph.ok:
        return ReconcileAttempt(peer, False, False, f"peer head fetch failed: {ph.error}", 0, 0, local_before, db.height, False, 0, [ph.error])
    if ph.chain_id != db.cfg.chain_id:
        return ReconcileAttempt(peer, False, False, "chain_id mismatch", 0, ph.height, local_before, db.height, False, 0, ["chain_id mismatch"])
    if ph.height <= 0:
        return ReconcileAttempt(peer, True, False, "peer has no receipts", 0, ph.height, local_before, db.height, False, 0, [])

    try:
        anc = find_common_ancestor(db, peer, ph.height)
    except Exception as e:
        return ReconcileAttempt(peer, False, False, "ancestor search failed", 0, ph.height, local_before, db.height, False, 0, [str(e)])

    if ph.height <= anc:
        return ReconcileAttempt(peer, True, False, "peer does not extend beyond common ancestor", anc, ph.height, local_before, db.height, False, 0, [])

    local_receipts, local_txs = build_local_maps_upto(db, anc)

    start = anc + 1
    merged_receipts = dict(local_receipts)
    merged_txs = dict(local_txs)

    try:
        while start <= ph.height:
            lim = min(pull_chunk, ph.height - start + 1)
            recs, txs = pull_peer_full(peer, start, lim)
            merged_receipts, merged_txs = merge_with_peer_suffix(merged_receipts, merged_txs, recs, txs, start)
            while start in merged_receipts and start in merged_txs:
                start += 1
            if lim <= 0:
                break
            if start == anc + 1 and (anc + 1 not in merged_receipts):
                raise TxReject("no progress pulling peer suffix")
    except Exception as e:
        errors.append(str(e))
        return ReconcileAttempt(peer, False, False, "pull failed", anc, ph.height, local_before, db.height, False, 0, errors)

    max_h = max(merged_receipts.keys(), default=anc)
    snap_path = db.best_snapshot_at_or_below(anc)

    ok, scanned, verified, ch, croot, crcpt, state, snap_used, snap_h, _, rep_errors = \
        full_replay_from_logs_with_optional_snapshot(merged_receipts, merged_txs, db.cfg, snap_path, max_h)

    if not ok:
        return ReconcileAttempt(peer, False, False, "candidate replay invalid", anc, ph.height, local_before, db.height, snap_used, snap_h, rep_errors)

    local_head_hex = hexs(db.head_receipt_hash)
    cand_head_hex = hexs(crcpt)
    better = fork_choice_better(ch, cand_head_hex, db.height, local_head_hex)

    if better:
        try:
            adopt_chain_atomic(db, merged_receipts, merged_txs, state, ch, croot, crcpt)
            return ReconcileAttempt(peer, True, True, f"adopted candidate (verified={verified}/{scanned})", anc, ph.height, local_before, db.height, snap_used, snap_h, [])
        except Exception as e:
            return ReconcileAttempt(peer, False, False, "adopt failed", anc, ph.height, local_before, db.height, snap_used, snap_h, [str(e)])

    return ReconcileAttempt(peer, True, False, "candidate valid but not better by fork-choice", anc, ph.height, local_before, db.height, snap_used, snap_h, [])

def reconcile_best(db: NodeDB, peers: List[str], pull_chunk: int = DEFAULT_MAX_PULL) -> Dict[str, Any]:
    local_verify = full_replay_db(db, commit=False)
    local_valid = bool(local_verify.ok and local_verify.matches_db)

    heads: List[PeerHead] = [fetch_peer_head(p) for p in peers]
    good_heads = [h for h in heads if h.ok and h.chain_id == db.cfg.chain_id and h.height > 0]
    good_heads.sort(key=lambda h: (h.height, h.head_receipt_hash), reverse=True)

    attempts: List[ReconcileAttempt] = []
    adopted = False

    for h in good_heads:
        if adopted:
            break
        if local_valid and not fork_choice_better(h.height, h.head_receipt_hash, db.height, hexs(db.head_receipt_hash)):
            continue
        att = reconcile_with_peer(db, h.peer, pull_chunk=pull_chunk)
        attempts.append(att)
        if att.adopted:
            adopted = True
            break

    if (not local_valid) and (not adopted):
        for h in good_heads:
            if adopted:
                break
            att = reconcile_with_peer(db, h.peer, pull_chunk=pull_chunk)
            attempts.append(att)
            if att.adopted:
                adopted = True
                break

    return {
        "ok": True,
        "local_valid": local_valid,
        "local_verify": dataclasses.asdict(local_verify),
        "peers": [dataclasses.asdict(x) for x in heads],
        "attempts": [dataclasses.asdict(a) for a in attempts],
        "adopted": adopted,
        "final_height": int(db.height),
        "final_root": hexs(db.root),
        "final_head_receipt_hash": hexs(db.head_receipt_hash),
    }

# ----------------------------- RPC Server --------------------------------------------

def parse_qs(qs: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not qs:
        return out
    for part in qs.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        out[k] = v
    return out

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

class RpcHandler(BaseHTTPRequestHandler):
    server_version = "MA_DELTA_V1.6_RPC"
    db: NodeDB

    def _send(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self, limit: int = 2_000_000) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        if n < 0 or n > limit:
            raise ValueError("body too large")
        return self.rfile.read(n)

    def do_GET(self) -> None:
        try:
            path = self.path.split("?", 1)[0]
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            db = self.db

            if path == "/health":
                with db._lock:
                    pol = db.cfg.policy
                    self._send(200, {
                        "ok": True,
                        "chain_id": str(int(db.cfg.chain_id)),
                        "height": int(db.height),
                        "root": hexs(db.root),
                        "head_receipt_hash": hexs(db.head_receipt_hash),
                        "snapshot_every": int(db.cfg.snapshot_every),
                        "keep_snapshots": int(db.cfg.keep_snapshots),
                        "require_signed": bool(db.cfg.require_signed),
                        "safe_push_full": bool(db.cfg.safe_push_full),
                        "eta_cap_q0_32": str(int(db.cfg.eta_cap_q0_32)),
                        "fork_choice": "height_then_headhash_lex",
                        "policy": {
                            "threshold": int(pol.threshold),
                            "allowlist_len": int(len(pol.normalized_allowset())),
                            "evidence_rules_len": int(len(pol.evidence_rules)),
                        },
                        "domains": {
                            "TX": DOMAIN_TX.decode("utf-8", "replace"),
                            "LEAF": DOMAIN_MERKLE_LEAF.decode("utf-8", "replace"),
                            "NODE": DOMAIN_MERKLE_NODE.decode("utf-8", "replace"),
                            "TXID": DOMAIN_TXID.decode("utf-8", "replace"),
                            "SIGPACK": DOMAIN_SIGPACK.decode("utf-8", "replace"),
                            "RCPT": DOMAIN_RCPT_HASH.decode("utf-8", "replace"),
                        },
                        "v": "1.6",
                    })
                return

            if path == "/metrics":
                with db._lock:
                    snap_latest = db.latest_snapshot()
                    self._send(200, {
                        "ok": True,
                        "chain_id": str(int(db.cfg.chain_id)),
                        "height": int(db.height),
                        "root": hexs(db.root),
                        "head_receipt_hash": hexs(db.head_receipt_hash),
                        "files": {
                            "state_bytes": file_size(db.state_path),
                            "policy_bytes": file_size(db.policy_path),
                            "receipts_bytes": file_size(db.receipts_path),
                            "txlog_bytes": file_size(db.txlog_path),
                        },
                        "snapshots": {
                            "latest": snap_latest or "",
                            "count": len([fn for fn in os.listdir(db.snap_dir) if fn.startswith("snap_")]),
                        },
                        "policy": {
                            "schema": db.cfg.policy.schema,
                            "chain_id": db.cfg.policy.chain_id,
                            "threshold": db.cfg.policy.threshold,
                            "allowlist_len": len(db.cfg.policy.normalized_allowset()),
                            "evidence_rules": [dataclasses.asdict(r) for r in db.cfg.policy.evidence_rules],
                        },
                        "config": {
                            "snapshot_every": int(db.cfg.snapshot_every),
                            "keep_snapshots": int(db.cfg.keep_snapshots),
                            "require_signed": bool(db.cfg.require_signed),
                            "safe_push_full": bool(db.cfg.safe_push_full),
                            "eta_cap_q0_32": str(int(db.cfg.eta_cap_q0_32)),
                            "max_submit_bytes": int(db.cfg.max_submit_bytes),
                        },
                    })
                return

            if path == "/policy":
                with db._lock:
                    self._send(200, {"ok": True, "policy": {
                        "schema": db.cfg.policy.schema,
                        "chain_id": db.cfg.policy.chain_id,
                        "threshold": db.cfg.policy.threshold,
                        "allowlist": db.cfg.policy.allowlist,
                        "evidence_rules": [dataclasses.asdict(r) for r in db.cfg.policy.evidence_rules],
                    }})
                return

            if path == "/state/root":
                with db._lock:
                    self._send(200, {"ok": True, "height": int(db.height), "root": hexs(db.root)})
                return

            if path == "/state":
                with db._lock:
                    self._send(200, {"ok": True, "dim": DIM, "q": "Q16.16", "height": int(db.height),
                                     "root": hexs(db.root), "state_q16_16": db.state_q16_16})
                return

            if path == "/receipt/head":
                with db._lock:
                    self._send(200, {"ok": True, "height": int(db.height),
                                     "head_receipt_hash": hexs(db.head_receipt_hash), "root": hexs(db.root)})
                return

            m = re.match(r"^/receipt/(\d+)$", path)
            if m:
                h = int(m.group(1))
                r = db.get_receipt_by_height(h)
                if r is None:
                    self._send(404, {"ok": False, "error": "not found"})
                else:
                    self._send(200, {"ok": True, "receipt": r})
                return

            m = re.match(r"^/receipt/hash/(\d+)$", path)
            if m:
                h = int(m.group(1))
                if h == 0:
                    self._send(200, {"ok": True, "height": 0, "receipt_hash": "00"*32})
                    return
                rh = db.get_receipt_hash(h)
                if rh is None:
                    self._send(404, {"ok": False, "error": "not found"})
                else:
                    self._send(200, {"ok": True, "height": h, "receipt_hash": hexs(rh)})
                return

            if path == "/receipt/range":
                params = parse_qs(qs)
                start = max(1, int(params.get("from", "1")))
                limit = max(1, min(int(params.get("limit", "50")), DEFAULT_MAX_RANGE))
                receipts = db.get_receipts_range(start, limit)
                self._send(200, {"ok": True, "from": start, "limit": limit, "receipts": receipts})
                return

            m = re.match(r"^/evidence/(\d+)$", path)
            if m:
                h = int(m.group(1))
                r = db.get_receipt_by_height(h)
                if r is None:
                    self._send(404, {"ok": False, "error": "not found"})
                else:
                    self._send(200, {"ok": True, "height": h,
                                     "evidence_type": int(r.get("evidence_type", 0)),
                                     "evidence_hash": str(r.get("evidence_hash", ""))})
                return

            m = re.match(r"^/proof/(\d+)$", path)
            if m:
                idx = int(m.group(1))
                if not (0 <= idx < DIM):
                    self._send(400, {"ok": False, "error": "index out of range"})
                    return
                with db._lock:
                    v = db.state_q16_16[idx]
                    pr = merkle_proof_macv1(db.state_q16_16, idx)
                    self._send(200, {"ok": True, "height": int(db.height), "root": hexs(db.root),
                                     "index": idx, "value_q16_16": int(v), "proof": [hexs(x) for x in pr]})
                return

            m = re.match(r"^/tx/(\d+)$", path)
            if m:
                h = int(m.group(1))
                t = db.get_tx_by_height(h)
                if t is None:
                    self._send(404, {"ok": False, "error": "not found"})
                else:
                    self._send(200, {"ok": True, "tx": t})
                return

            if path == "/tx/range":
                params = parse_qs(qs)
                start = max(1, int(params.get("from", "1")))
                limit = max(1, min(int(params.get("limit", "50")), DEFAULT_MAX_RANGE))
                txs = db.get_txs_range(start, limit)
                self._send(200, {"ok": True, "from": start, "limit": limit, "txs": txs})
                return

            self._send(404, {"ok": False, "error": "unknown endpoint"})
        except TxReject as e:
            self._send(400, {"ok": False, "error": str(e)})
        except Exception as e:
            self._send(500, {"ok": False, "error": f"server error: {e}"})

    def do_POST(self) -> None:
        try:
            path = self.path.split("?", 1)[0]
            db = self.db

            if path == "/submit":
                body = self._read_body(limit=2_000_000)
                req = json.loads(body.decode("utf-8")) if body else {}
                tx_hex = req.get("tx_hex", "")
                tx_b64 = req.get("tx_b64", "")
                if tx_hex:
                    tx_bytes = bytes.fromhex(tx_hex)
                elif tx_b64:
                    tx_bytes = b64d(tx_b64)
                else:
                    raise TxReject("provide tx_hex or tx_b64")
                out = apply_tx_to_db(db, tx_bytes)
                self._send(200, out)
                return

            if path == "/sync/heads":
                with db._lock:
                    self._send(200, {"ok": True, "chain_id": str(int(db.cfg.chain_id)),
                                     "height": int(db.height), "head_receipt_hash": hexs(db.head_receipt_hash),
                                     "root": hexs(db.root)})
                return

            if path == "/sync/pull_full":
                body = self._read_body(limit=1_000_000)
                req = json.loads(body.decode("utf-8")) if body else {}
                start = max(1, int(req.get("from_height", 1)))
                limit = max(1, min(int(req.get("limit", 50)), DEFAULT_MAX_PULL))
                receipts = db.get_receipts_range(start, limit)
                txs = db.get_txs_range(start, limit)
                self._send(200, {"ok": True, "from_height": start, "limit": limit, "receipts": receipts, "txs": txs})
                return

            if path == "/sync/push_full":
                body = self._read_body(limit=3_000_000)
                req = json.loads(body.decode("utf-8")) if body else {}
                recs = req.get("receipts", [])
                txs = req.get("txs", [])
                require(isinstance(recs, list), "receipts must be list")
                require(isinstance(txs, list), "txs must be list")

                in_tx: Dict[int, Dict[str, Any]] = {}
                for t in txs:
                    if isinstance(t, dict) and t.get("schema") in ("MA_TXLOG_V1.5", "MA_TXLOG_V1.6"):
                        try:
                            in_tx[int(t.get("height", -1))] = t
                        except Exception:
                            pass

                accepted = 0
                snap_path = ""

                with db._lock:
                    expected_h = db.height + 1
                    prev_rcpt = db.head_receipt_hash
                    db.root = merkle_root_macv1(db.state_q16_16)

                    for r in recs:
                        if not isinstance(r, dict):
                            continue
                        if "height" not in r or "receipt_hash" not in r or "txid" not in r:
                            continue

                        h = int(r.get("height", -1))
                        if h != expected_h:
                            break

                        t = in_tx.get(h)
                        require(t is not None, f"missing tx for pushed receipt height {h}")

                        tx_bytes = b64d(t["tx_b64"])
                        require(len(tx_bytes) <= db.cfg.max_submit_bytes, "pushed tx too large")

                        ptx = parse_tx_bytes(tx_bytes)
                        validate_tx_struct(ptx, db.cfg)
                        valid_signers = verify_sigs_and_policy(ptx, db.cfg)

                        evidence = parse_tlv_evidence(ptx.tlv)
                        validate_evidence_rules(evidence, valid_signers, db.cfg.policy)

                        txid_bytes = compute_txid(ptx.header_bytes, ptx.payload, ptx.sigpairs)
                        require(hexs(txid_bytes) == str(r.get("txid", "")), "pushed receipt txid mismatch (computed)")
                        require(str(t.get("txid", "")) == str(r.get("txid", "")), "pushed txlog txid mismatch to receipt")

                        require(str(r.get("prev_receipt_hash", "")) == hexs(prev_rcpt), "prev hash mismatch for push_full")
                        require(str(r.get("pre_root", "")) == hexs(db.root), "pre_root mismatch to local state")

                        targets = payload_to_targets_q16_16(ptx.targets_payload)
                        eta_max = int(ptx.header.eta_max_q0_32) if (ptx.header.flags & FLAG_ETA_CLAMP) else 0
                        apply = apply_perfect_t(
                            state_q16_16=db.state_q16_16,
                            targets_q16_16=targets,
                            alpha_q0_32=int(ptx.header.alpha_q0_32),
                            eps_q32_32=int(ptx.header.eps_q32_32),
                            eta_max_q0_32=eta_max,
                            do_clip=bool(ptx.header.flags & FLAG_CLIP),
                        )
                        gate_regression(apply, ptx.header, db.cfg)

                        require(str(r.get("post_root", "")) == hexs(apply.post_root), "post_root mismatch on push_full")
                        require(str(r.get("pre_d2_q32_32", "")) == str(int(apply.pre_d2_q32_32)), "pre_d2 mismatch on push_full")
                        require(str(r.get("post_d2_q32_32", "")) == str(int(apply.post_d2_q32_32)), "post_d2 mismatch on push_full")
                        require(str(r.get("eta_q0_32", "")) == str(int(apply.eta_q0_32)), "eta mismatch on push_full")

                        exp_ev_type = int(evidence.evidence_type if evidence.present else 0)
                        exp_ev_hash = evidence.evidence_hash if evidence.present else (b"\x00"*32)
                        got_ev_type = int(r.get("evidence_type", 0)) if "evidence_type" in r else 0
                        got_ev_hash = unhex(r.get("evidence_hash", "00"*64)) if "evidence_hash" in r else (b"\x00"*32)
                        require(got_ev_type == exp_ev_type, "evidence_type mismatch on push_full")
                        require(got_ev_hash == exp_ev_hash, "evidence_hash mismatch on push_full")

                        if "valid_signers" in r:
                            rs = r.get("valid_signers", [])
                            if isinstance(rs, list):
                                rs_set = set(str(x) for x in rs)
                                require(rs_set.issubset(set(valid_signers)), "valid_signers mismatch on push_full")

                        core = make_receipt_core(
                            chain_id=db.cfg.chain_id,
                            height=h,
                            prev_receipt_hash=prev_rcpt,
                            txid_bytes=txid_bytes,
                            pre_root=apply.pre_root,
                            post_root=apply.post_root,
                            pre_d2_q32_32=apply.pre_d2_q32_32,
                            post_d2_q32_32=apply.post_d2_q32_32,
                            eta_q0_32=apply.eta_q0_32,
                            nonce=int(r.get("nonce", 0)),
                            timestamp_ms=int(r.get("timestamp_ms", 0)),
                            evidence_type=exp_ev_type,
                            evidence_hash32=(exp_ev_hash if evidence.present else b""),
                        )
                        rh = receipt_hash(core)
                        require(hexs(rh) == str(r.get("receipt_hash", "")), "receipt hash mismatch for push_full")

                        db.append_txlog(t)
                        db.append_receipt(r)
                        db.root = apply.post_root

                        prev_rcpt = rh
                        expected_h += 1
                        accepted += 1

                    db.save_state()
                    if db.cfg.snapshot_every > 0 and (db.height % db.cfg.snapshot_every == 0):
                        snap_path = db.write_snapshot()

                self._send(200, {"ok": True, "accepted": accepted, "new_height": int(db.height),
                                 "head_receipt_hash": hexs(db.head_receipt_hash), "root": hexs(db.root),
                                 "snapshot": snap_path})
                return

            if path == "/admin/verify":
                body = self._read_body(limit=500_000)
                req = json.loads(body.decode("utf-8")) if body else {}
                commit = bool(req.get("commit", False))
                res = full_replay_db(db, commit=commit)
                self._send(200, dataclasses.asdict(res))
                return

            if path == "/admin/reconcile":
                body = self._read_body(limit=1_000_000)
                req = json.loads(body.decode("utf-8")) if body else {}
                peers = req.get("peers", [])
                pull_chunk = int(req.get("pull_chunk", DEFAULT_MAX_PULL))
                if not isinstance(peers, list) or not all(isinstance(p, str) for p in peers):
                    raise TxReject("peers must be list[str]")
                pull_chunk = max(1, min(pull_chunk, 500))
                res = reconcile_best(db, peers=peers, pull_chunk=pull_chunk)
                self._send(200, res)
                return

            self._send(404, {"ok": False, "error": "unknown endpoint"})
        except TxReject as e:
            self._send(400, {"ok": False, "error": str(e)})
        except Exception as e:
            self._send(500, {"ok": False, "error": f"server error: {e}"})

    def log_message(self, fmt: str, *args) -> None:
        return

def run_server(db: NodeDB, host: str, port: int) -> None:
    def handler(*args, **kwargs):
        h = RpcHandler(*args, **kwargs)
        h.db = db
        return h
    srv = ThreadingHTTPServer((host, port), handler)
    print(f"[rpc] http://{host}:{port} chain_id={db.cfg.chain_id} height={db.height} root={hexs(db.root)[:16]}… v1.6 thr={db.cfg.policy.threshold} allow={len(db.cfg.policy.normalized_allowset())} rules={len(db.cfg.policy.evidence_rules)}")
    srv.serve_forever()

# ----------------------------- CLI: TLV + TX Creation --------------------------------

def tlv_encode(items: List[Tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for t, v in items:
        if not (0 <= t <= 255):
            raise ValueError("TLV type out of range")
        if len(v) > 65535:
            raise ValueError("TLV value too large")
        out.append(t & 0xFF)
        out += int(len(v)).to_bytes(2, "little")
        out += v
    return bytes(out)

def evidence_hash_from_file(path: str) -> bytes:
    data = read_file(path)
    return sha256(data)

def make_tx_bytes(
    chain_id: int,
    nonce: int,
    timestamp_ms: int,
    alpha_q0_32: int,
    eps_q32_32: int,
    eta_max_q0_32: int,
    clip: bool,
    sign: bool,
    seed32: Optional[bytes],
    extra_seeds32: List[bytes],
    target_const: float,
    target_file: str,
    evidence_file: str,
    evidence_type: int,
) -> bytes:
    if target_file:
        tf = read_json(target_file)
        if not isinstance(tf, list) or len(tf) != DIM:
            raise ValueError("target-file must be JSON list length 256")
        t_q8_8: List[int] = []
        for v in tf:
            enc = int(round(float(v) * 256.0))
            if enc < -32768 or enc > 32767:
                raise ValueError("target value out of int16 Q8.8 range")
            t_q8_8.append(enc)
    else:
        enc = int(round(float(target_const) * 256.0))
        if enc < -32768 or enc > 32767:
            raise ValueError("target constant out of int16 Q8.8 range")
        t_q8_8 = [enc] * DIM

    targets_payload = b"".join(struct.pack("<h", int(v)) for v in t_q8_8)

    tlv_items: List[Tuple[int, bytes]] = []
    flags = 0
    if clip:
        flags |= FLAG_CLIP
    if eta_max_q0_32 != 0:
        flags |= FLAG_ETA_CLAMP

    if evidence_file:
        eh = evidence_hash_from_file(evidence_file)
        et = int(evidence_type) & 0xFF
        tlv_items.append((TLV_EVIDENCE, bytes([et]) + eh))
        flags |= FLAG_HAS_TLV

    tlv_blob = tlv_encode(tlv_items) if tlv_items else b""
    payload = targets_payload + tlv_blob

    if sign:
        flags |= FLAG_INCLUDE_PUBKEY
        if extra_seeds32:
            flags |= FLAG_MULTI_SIG

    h = MaDeltaHeaderV1(
        magic=MAGIC,
        version=VERSION,
        dim=DIM,
        op=OP_PERFECT_T,
        flags=flags,
        reserved0=0,
        alpha_q0_32=u32(alpha_q0_32),
        eps_q32_32=u64(eps_q32_32),
        eta_max_q0_32=u32(eta_max_q0_32),
        nonce=u32(nonce),
        timestamp_ms=u64(timestamp_ms),
        chain_id=u64(chain_id),
        payload_len=u32(len(payload)),
        reserved1=b"\x00" * 12,
    )
    header_bytes = h.pack()
    txb = header_bytes + payload

    sigpairs: List[SigPair] = []
    if sign:
        if seed32 is None or len(seed32) != 32:
            raise ValueError("seed required for sign (32 bytes)")
        msg = tx_message(chain_id, header_bytes, payload)

        sk, pk = ed25519_keypair(seed32)
        sig = ed25519_sign(sk, msg)
        sigpairs.append(SigPair(pk, sig))

        for s in extra_seeds32:
            if len(s) != 32:
                raise ValueError("extra seed must be 32 bytes")
            sk2, pk2 = ed25519_keypair(s)
            sig2 = ed25519_sign(sk2, msg)
            sigpairs.append(SigPair(pk2, sig2))

        txb += sigpairs[0].pubkey + sigpairs[0].sig
        if extra_seeds32:
            txb += bytes([len(sigpairs) - 1])
            for sp in sigpairs[1:]:
                txb += sp.pubkey + sp.sig

    return txb

# ----------------------------- Init / Policy Commands --------------------------------

def init_dir(data_dir: str, chain_id: int) -> None:
    ensure_dir(data_dir)
    cfg = default_config(chain_id, data_dir)
    db = NodeDB(data_dir, cfg)
    db.save_state()
    db.write_snapshot()
    if not os.path.exists(db.receipts_path):
        write_file(db.receipts_path, b"")
    if not os.path.exists(db.txlog_path):
        write_file(db.txlog_path, b"")
    _ = load_policy(data_dir, chain_id)
    print(f"Initialized {data_dir} chain_id={chain_id} height={db.height} root={hexs(db.root)} policy={os.path.join(data_dir, POLICY_JSON)}")

def cmd_init(args: argparse.Namespace) -> None:
    init_dir(args.dir, args.chain_id)

def cmd_policy_set(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    allow: List[str] = []
    if args.allowlist:
        txt = read_file(args.allowlist).decode("utf-8", "replace")
        for line in txt.splitlines():
            line = line.strip().lower()
            if not line:
                continue
            if re.fullmatch(r"[0-9a-f]{64}", line):
                allow.append(line)
    pol = Policy(schema="MA_POLICY_V1.6", chain_id=str(int(args.chain_id)),
                 threshold=max(1, int(args.threshold)), allowlist=allow, evidence_rules=[])
    save_policy(args.dir, pol)
    print("Wrote policy:", os.path.join(args.dir, POLICY_JSON))
    print(json.dumps({
        "schema": pol.schema,
        "chain_id": pol.chain_id,
        "threshold": pol.threshold,
        "allowlist": pol.allowlist,
        "evidence_rules": [dataclasses.asdict(r) for r in pol.evidence_rules],
    }, indent=2, sort_keys=True))

def cmd_policy_add(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    # best-effort chain_id inference: policy itself is authoritative
    pol = load_policy(args.dir, chain_id=1)
    pk = args.pubkey_hex.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", pk):
        raise SystemExit("pubkey-hex must be 64 hex chars")
    aset = pol.normalized_allowset()
    if pk not in aset:
        pol.allowlist.append(pk)
    save_policy(args.dir, pol)
    print("Updated policy:", os.path.join(args.dir, POLICY_JSON))
    print(json.dumps({
        "schema": pol.schema,
        "chain_id": pol.chain_id,
        "threshold": pol.threshold,
        "allowlist": pol.allowlist,
        "evidence_rules": [dataclasses.asdict(r) for r in pol.evidence_rules],
    }, indent=2, sort_keys=True))

def cmd_policy_add_rule(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    pol = load_policy(args.dir, chain_id=1)
    et = int(args.evidence_type)
    ms = max(1, int(args.min_signers))
    reqs: List[str] = []
    if args.required_signers:
        for s in args.required_signers.split(","):
            s = s.strip().lower()
            if not s:
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", s):
                raise SystemExit(f"required signer not 64-hex pubkey: {s[:16]}…")
            reqs.append(s)
    # upsert by evidence_type
    new_rules = [r for r in pol.evidence_rules if int(r.evidence_type) != et]
    new_rules.append(EvidenceRule(evidence_type=et, min_signers=ms, required_signers=reqs))
    new_rules.sort(key=lambda r: r.evidence_type)
    pol.evidence_rules = new_rules
    save_policy(args.dir, pol)
    print("Updated policy rule:", os.path.join(args.dir, POLICY_JSON))
    print(json.dumps({
        "schema": pol.schema,
        "chain_id": pol.chain_id,
        "threshold": pol.threshold,
        "allowlist": pol.allowlist,
        "evidence_rules": [dataclasses.asdict(r) for r in pol.evidence_rules],
    }, indent=2, sort_keys=True))

# ----------------------------- Run / Submit / Verify / Reconcile ---------------------

def cmd_run(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    cfg = default_config(args.chain_id, args.dir)
    cfg.snapshot_every = args.snapshot_every
    cfg.keep_snapshots = args.keep_snapshots
    cfg.require_signed = not args.allow_unsigned
    cfg.eta_cap_q0_32 = int(args.eta_cap_q0_32)
    cfg.max_submit_bytes = int(args.max_submit_bytes)
    cfg.safe_push_full = not args.unsafe_push_full
    cfg.policy = load_policy(args.dir, args.chain_id)
    db = NodeDB(args.dir, cfg)
    run_server(db, args.host, args.port)

def cmd_make_tx(args: argparse.Namespace) -> None:
    seed32 = bytes.fromhex(args.seed_hex) if args.seed_hex else None
    extra_seeds32: List[bytes] = []
    if args.extra_seeds:
        for s in args.extra_seeds.split(","):
            s = s.strip()
            if not s:
                continue
            extra_seeds32.append(bytes.fromhex(s))

    if args.sign and (seed32 is None or len(seed32) != 32):
        raise SystemExit("When --sign, provide --seed-hex (32 bytes hex).")
    ts = args.timestamp_ms if args.timestamp_ms else now_ms()

    txb = make_tx_bytes(
        chain_id=args.chain_id,
        nonce=args.nonce,
        timestamp_ms=ts,
        alpha_q0_32=int(args.alpha_q0_32),
        eps_q32_32=int(args.eps_q32_32),
        eta_max_q0_32=int(args.eta_max_q0_32),
        clip=bool(args.clip),
        sign=bool(args.sign),
        seed32=seed32,
        extra_seeds32=extra_seeds32,
        target_const=float(args.target),
        target_file=args.target_file,
        evidence_file=args.evidence_file,
        evidence_type=int(args.evidence_type),
    )
    write_file(args.out, txb)
    print(f"Wrote tx {args.out} bytes={len(txb)}")

def cmd_submit(args: argparse.Namespace) -> None:
    txb = read_file(args.tx)
    status, j = http_post_json(args.url, "/submit", {"tx_b64": b64e(txb)})
    print("status:", status)
    print(json.dumps(j, indent=2, sort_keys=True))

def cmd_verify(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    cfg = default_config(args.chain_id, args.dir)
    cfg.policy = load_policy(args.dir, args.chain_id)
    db = NodeDB(args.dir, cfg)
    res = full_replay_db(db, commit=bool(args.commit))
    print(json.dumps(dataclasses.asdict(res), indent=2, sort_keys=True))

def cmd_reconcile(args: argparse.Namespace) -> None:
    ensure_dir(args.dir)
    cfg = default_config(args.chain_id, args.dir)
    cfg.policy = load_policy(args.dir, args.chain_id)
    db = NodeDB(args.dir, cfg)
    peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    res = reconcile_best(db, peers=peers, pull_chunk=int(args.pull_chunk))
    print(json.dumps(res, indent=2, sort_keys=True))

# ----------------------------- DEMOS (v1.6) ------------------------------------------

def _tmp_clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    ensure_dir(path)

def cmd_demo_router_3of5(_: argparse.Namespace) -> None:
    """
    Demo: "use-case snippet: fusing 256-dim router weights from 5 operators with 3-of-5 approval"
      - Creates a temp chain with threshold=3 and 5 allowlisted keys
      - Submits a single 3-signature tx to move vector toward +1.0 target
      - Prints receipt head + a merkle proof for index 0
    """
    d = ".ma_demo_router_3of5"
    _tmp_clean_dir(d)
    chain_id = 7777
    init_dir(d, chain_id)

    # build 5 deterministic keys
    seeds = [bytes([i]) * 32 for i in (11, 22, 33, 44, 55)]
    pubs = [ed25519_keypair(s)[1] for s in seeds]
    pol = Policy(schema="MA_POLICY_V1.6", chain_id=str(chain_id), threshold=3,
                 allowlist=[hexs(pk) for pk in pubs], evidence_rules=[])
    save_policy(d, pol)

    cfg = default_config(chain_id, d)
    cfg.policy = load_policy(d, chain_id)
    cfg.require_signed = True
    cfg.snapshot_every = 1
    db = NodeDB(d, cfg)

    tx = make_tx_bytes(
        chain_id=chain_id,
        nonce=1,
        timestamp_ms=now_ms(),
        alpha_q0_32=DEFAULT_ALPHA_Q0_32,
        eps_q32_32=DEFAULT_EPS_Q32_32,
        eta_max_q0_32=int(1.5 * (1 << 32)),
        clip=False,
        sign=True,
        seed32=seeds[0],
        extra_seeds32=[seeds[1], seeds[2]],  # 3 total signers
        target_const=1.0,
        target_file="",
        evidence_file="",
        evidence_type=0,
    )

    out = apply_tx_to_db(db, tx)
    print("=== DEMO router 3-of-5 ===")
    print(json.dumps(out, indent=2, sort_keys=True))
    pr = merkle_proof_macv1(db.state_q16_16, 0)
    print("proof(index=0):", [hexs(x) for x in pr][:3], "…", f"(len={len(pr)})")
    print("state[0] q16.16:", db.state_q16_16[0])

def cmd_demo_evidence_oracle(_: argparse.Namespace) -> None:
    """
    Demo: "evidence-bound update requiring oracle signer"
      - threshold=2 allowlist of {oracle, op1, op2}
      - evidence rule: evidence_type=7 must include oracle among signers (min_signers=1 from required_signers={oracle})
      - (A) valid tx: (oracle + op1) signs evidence_type=7 update -> ACCEPT
      - (B) invalid tx: (op1 + op2) signs evidence_type=7 update -> REJECT
    """
    d = ".ma_demo_evidence_oracle"
    _tmp_clean_dir(d)
    chain_id = 8888
    init_dir(d, chain_id)

    oracle_seed = b"O" * 32
    op1_seed = b"A" * 32
    op2_seed = b"B" * 32
    oracle_pk = ed25519_keypair(oracle_seed)[1]
    op1_pk = ed25519_keypair(op1_seed)[1]
    op2_pk = ed25519_keypair(op2_seed)[1]

    pol = Policy(schema="MA_POLICY_V1.6", chain_id=str(chain_id), threshold=2,
                 allowlist=[hexs(oracle_pk), hexs(op1_pk), hexs(op2_pk)],
                 evidence_rules=[EvidenceRule(evidence_type=7, min_signers=1, required_signers=[hexs(oracle_pk)])])
    save_policy(d, pol)

    cfg = default_config(chain_id, d)
    cfg.policy = load_policy(d, chain_id)
    cfg.require_signed = True
    cfg.snapshot_every = 1
    db = NodeDB(d, cfg)

    ev_path = os.path.join(d, "evidence.bin")
    write_file(ev_path, b"demo evidence payload v1")

    tx_ok = make_tx_bytes(
        chain_id=chain_id, nonce=1, timestamp_ms=now_ms(),
        alpha_q0_32=DEFAULT_ALPHA_Q0_32, eps_q32_32=DEFAULT_EPS_Q32_32, eta_max_q0_32=int(1.5*(1<<32)),
        clip=False, sign=True, seed32=oracle_seed, extra_seeds32=[op1_seed],
        target_const=0.5, target_file="", evidence_file=ev_path, evidence_type=7
    )
    print("=== DEMO evidence oracle: valid tx (oracle+op1) ===")
    print(json.dumps(apply_tx_to_db(db, tx_ok), indent=2, sort_keys=True))

    tx_bad = make_tx_bytes(
        chain_id=chain_id, nonce=2, timestamp_ms=now_ms(),
        alpha_q0_32=DEFAULT_ALPHA_Q0_32, eps_q32_32=DEFAULT_EPS_Q32_32, eta_max_q0_32=int(1.5*(1<<32)),
        clip=False, sign=True, seed32=op1_seed, extra_seeds32=[op2_seed],
        target_const=0.25, target_file="", evidence_file=ev_path, evidence_type=7
    )
    print("=== DEMO evidence oracle: invalid tx (op1+op2, missing oracle) ===")
    try:
        apply_tx_to_db(db, tx_bad)
        print("ERROR: expected rejection but tx applied")
    except TxReject as e:
        print("Rejected as expected:", str(e))

# ----------------------------- SELFTEST ----------------------------------------------

def cmd_selftest(_: argparse.Namespace) -> None:
    seed1 = bytes.fromhex("11" * 32)
    seed2 = bytes.fromhex("22" * 32)
    seed3 = bytes.fromhex("33" * 32)
    _, pk1 = ed25519_keypair(seed1)
    _, pk2 = ed25519_keypair(seed2)
    _, pk3 = ed25519_keypair(seed3)

    a_dir = ".ma_tmp_a_v16"
    b_dir = ".ma_tmp_b_v16"
    for d in (a_dir, b_dir):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        init_dir(d, chain_id=999)

    # policy: threshold=2 of {pk1,pk2,pk3}; evidence rule type=7 requires pk1
    pol = Policy(schema="MA_POLICY_V1.6", chain_id="999", threshold=2,
                 allowlist=[hexs(pk1), hexs(pk2), hexs(pk3)],
                 evidence_rules=[EvidenceRule(evidence_type=7, min_signers=1, required_signers=[hexs(pk1)])])
    save_policy(a_dir, pol)
    save_policy(b_dir, pol)

    cfgA = default_config(999, a_dir); cfgA.require_signed = True; cfgA.snapshot_every = 2; cfgA.policy = load_policy(a_dir, 999)
    cfgB = default_config(999, b_dir); cfgB.require_signed = True; cfgB.snapshot_every = 2; cfgB.policy = load_policy(b_dir, 999)
    dbA = NodeDB(a_dir, cfgA)
    dbB = NodeDB(b_dir, cfgB)

    ev_path = os.path.join(a_dir, "ev.bin")
    write_file(ev_path, b"evidence blob")

    # good tx with evidence type 7 signed by pk1 + pk2
    tx1 = make_tx_bytes(999, 1, now_ms(), DEFAULT_ALPHA_Q0_32, DEFAULT_EPS_Q32_32, int(1.5*(1<<32)),
                        False, True, seed1, [seed2], 0.0, "", ev_path, 7)
    # bad tx with evidence type 7 missing pk1 (pk2 + pk3) should reject
    tx_bad = make_tx_bytes(999, 2, now_ms(), DEFAULT_ALPHA_Q0_32, DEFAULT_EPS_Q32_32, int(1.5*(1<<32)),
                           False, True, seed2, [seed3], 0.0, "", ev_path, 7)

    apply_tx_to_db(dbA, tx1)
    apply_tx_to_db(dbB, tx1)

    try:
        apply_tx_to_db(dbA, tx_bad)
        raise AssertionError("expected reject")
    except TxReject:
        pass

    # fork B ahead with a valid non-evidence tx (pk1+pk3)
    tx2b = make_tx_bytes(999, 3, now_ms(), DEFAULT_ALPHA_Q0_32, DEFAULT_EPS_Q32_32, int(1.5*(1<<32)),
                         False, True, seed1, [seed3], 1.0, "", "", 0)
    apply_tx_to_db(dbB, tx2b)

    def serve(db: NodeDB, host: str, port: int):
        run_server(db, host, port)

    thA = threading.Thread(target=serve, args=(dbA, "127.0.0.1", 18290), daemon=True)
    thB = threading.Thread(target=serve, args=(dbB, "127.0.0.1", 18291), daemon=True)
    thA.start(); thB.start()
    time.sleep(0.25)

    res = reconcile_best(dbA, peers=["http://127.0.0.1:18291"], pull_chunk=50)
    assert res["ok"] and res["adopted"]
    assert dbA.height == dbB.height
    assert hexs(dbA.head_receipt_hash) == hexs(dbB.head_receipt_hash)

    vr = full_replay_db(dbA, commit=False)
    assert vr.ok and vr.matches_db

    print("SELFTEST: OK")

# ----------------------------- CLI ---------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ma_delta_v1_6.py")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_i = sp.add_parser("init", help="Initialize data dir")
    p_i.add_argument("--dir", required=True)
    p_i.add_argument("--chain-id", type=int, default=1)
    p_i.set_defaults(func=cmd_init)

    p_ps = sp.add_parser("policy-set", help="Write policy.json with threshold + allowlist file")
    p_ps.add_argument("--dir", required=True)
    p_ps.add_argument("--chain-id", type=int, default=1)
    p_ps.add_argument("--threshold", type=int, default=1)
    p_ps.add_argument("--allowlist", default="", help="text file of pubkey hex (one per line)")
    p_ps.set_defaults(func=cmd_policy_set)

    p_pa = sp.add_parser("policy-add", help="Add a pubkey hex to policy.json allowlist")
    p_pa.add_argument("--dir", required=True)
    p_pa.add_argument("--pubkey-hex", required=True)
    p_pa.set_defaults(func=cmd_policy_add)

    p_pr = sp.add_parser("policy-add-rule", help="Upsert an evidence rule by evidence_type")
    p_pr.add_argument("--dir", required=True)
    p_pr.add_argument("--evidence-type", type=int, required=True)
    p_pr.add_argument("--min-signers", type=int, default=1)
    p_pr.add_argument("--required-signers", default="", help="comma-separated pubkey hex; empty => use allowlist")
    p_pr.set_defaults(func=cmd_policy_add_rule)

    p_r = sp.add_parser("run", help="Run RPC server")
    p_r.add_argument("--dir", required=True)
    p_r.add_argument("--chain-id", type=int, default=1)
    p_r.add_argument("--host", default="127.0.0.1")
    p_r.add_argument("--port", type=int, default=8080)
    p_r.add_argument("--snapshot-every", type=int, default=DEFAULT_SNAPSHOT_EVERY)
    p_r.add_argument("--keep-snapshots", type=int, default=DEFAULT_KEEP_SNAPSHOTS)
    p_r.add_argument("--allow-unsigned", action="store_true")
    p_r.add_argument("--unsafe-push-full", action="store_true", help="Disable safe push_full validation+apply")
    p_r.add_argument("--eta-cap-q0-32", type=int, default=DEFAULT_ETA_CAP_Q0_32)
    p_r.add_argument("--max-submit-bytes", type=int, default=DEFAULT_MAX_SUBMIT_BYTES)
    p_r.set_defaults(func=cmd_run)

    p_m = sp.add_parser("make-tx", help="Create tx bytes (optional multisig + evidence)")
    p_m.add_argument("--out", required=True)
    p_m.add_argument("--chain-id", type=int, default=1)
    p_m.add_argument("--nonce", type=int, default=0)
    p_m.add_argument("--timestamp-ms", type=int, default=0)
    p_m.add_argument("--alpha-q0-32", dest="alpha_q0_32", type=int, default=DEFAULT_ALPHA_Q0_32)
    p_m.add_argument("--eps-q32-32", dest="eps_q32_32", type=int, default=DEFAULT_EPS_Q32_32)
    p_m.add_argument("--eta-max-q0-32", dest="eta_max_q0_32", type=int, default=0)
    p_m.add_argument("--clip", action="store_true")
    p_m.add_argument("--target", default="0.0")
    p_m.add_argument("--target-file", default="")
    p_m.add_argument("--evidence-file", default="", help="file to hash and bind into receipt via TLV")
    p_m.add_argument("--evidence-type", type=int, default=0)
    p_m.add_argument("--sign", action="store_true")
    p_m.add_argument("--seed-hex", default="", help="32-byte hex seed for primary signer")
    p_m.add_argument("--extra-seeds", default="", help="comma-separated 32-byte hex seeds for extra signatures")
    p_m.set_defaults(func=cmd_make_tx)

    p_s = sp.add_parser("submit", help="Submit tx to RPC server")
    p_s.add_argument("--url", required=True, help="http://host:port")
    p_s.add_argument("--tx", required=True)
    p_s.set_defaults(func=cmd_submit)

    p_v = sp.add_parser("verify", help="Snapshot-accelerated full replay and validate receipts; optional commit")
    p_v.add_argument("--dir", required=True)
    p_v.add_argument("--chain-id", type=int, default=1)
    p_v.add_argument("--commit", action="store_true")
    p_v.set_defaults(func=cmd_verify)

    p_c = sp.add_parser("reconcile", help="Reconcile vs peers using fork-choice; may adopt best chain")
    p_c.add_argument("--dir", required=True)
    p_c.add_argument("--chain-id", type=int, default=1)
    p_c.add_argument("--peers", required=True, help="comma-separated base urls: http://h1:8080,http://h2:8080")
    p_c.add_argument("--pull-chunk", type=int, default=DEFAULT_MAX_PULL)
    p_c.set_defaults(func=cmd_reconcile)

    p_d1 = sp.add_parser("demo-router-3of5", help="Demo: 3-of-5 approval updates a router vector")
    p_d1.set_defaults(func=cmd_demo_router_3of5)

    p_d2 = sp.add_parser("demo-evidence-oracle", help="Demo: evidence rule requiring oracle signer")
    p_d2.set_defaults(func=cmd_demo_evidence_oracle)

    p_t = sp.add_parser("selftest", help="Run internal tests (policy threshold + evidence rule + fork + reconcile)")
    p_t.set_defaults(func=cmd_selftest)

    return p

def main() -> None:
    p = build_cli()
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
