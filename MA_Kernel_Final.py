#!/usr/bin/env python3
# ======================================================================================
# MA KERNEL (Memory Arithmetic) — Single-File, GitHub/Colab Ready
# ======================================================================================
# What you get:
#   - Hash-chained DAG "values" (nodes) with canonical serialization (SHA-256 id)
#   - Small evaluator VM (CONST / arithmetic / lists / tuples / merkle / if / comparisons)
#   - Learn-pass clustering (signature-based) -> ABSTRACT nodes
#   - Minimal proof extraction (reachable subgraph needed to show ancestry)
#   - DOT export + (optional) render PNG/SVG via Graphviz
#   - Fully automated "auto" pipeline:
#       * Build demo state
#       * Learn pass
#       * Export DOT(s)
#       * Generate proofs
#       * GC prune snapshot
#       * Render images + write an HTML INDEX
#       * Generate charts (PNG) for quick run diagnostics
#       * Zip the whole run directory
#       * (optional) auto-download zip in Colab
#
# Colab quickstart (one cell):
#   !apt-get update -y && apt-get install -y graphviz
#   !python ma_kernel.py auto --colab-download
#
# Local quickstart:
#   python3 ma_kernel.py auto --out-dir ma_kernel_run_test
#
# ======================================================================================

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Bytes32 = bytes

# ----------------------------
# Small utilities
# ----------------------------

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _u32(n: int) -> bytes:
    if n < 0 or n > 0xFFFFFFFF:
        raise ValueError("u32 out of range")
    return struct.pack("<I", n)


def _u64(n: int) -> bytes:
    if n < 0 or n > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("u64 out of range")
    return struct.pack("<Q", n)


def _i64_pack(n: int) -> bytes:
    if n < -(1 << 63) or n > (1 << 63) - 1:
        raise OverflowError("i64 out of range")
    return struct.pack("<q", n)


def _i64_wrap(n: int) -> int:
    n &= (1 << 64) - 1
    if n >= (1 << 63):
        n -= 1 << 64
    return n


def _is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def _in_ipython() -> bool:
    try:
        import IPython  # noqa
        return True
    except Exception:
        return False


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", ".", ":", "@", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


# ----------------------------
# Canonical node encoding
# ----------------------------

def canonical_node_bytes(op: str, parents: Sequence[bytes], payload: bytes, meta_commit: bytes) -> bytes:
    op_b = op.encode("utf-8")
    if len(op_b) > 255:
        raise ValueError("opcode too long")
    if any(len(p) != 32 for p in parents):
        raise ValueError("parent hash must be 32 bytes")
    if len(payload) > 10_000_000:
        raise ValueError("payload too large")
    if len(meta_commit) > 1_000_000:
        raise ValueError("meta_commit too large")

    parts = []
    parts.append(b"MAK0")
    parts.append(struct.pack("B", len(op_b)))
    parts.append(op_b)
    parts.append(_u32(len(parents)))
    for p in parents:
        parts.append(p)
    parts.append(_u32(len(payload)))
    parts.append(payload)
    parts.append(_u32(len(meta_commit)))
    parts.append(meta_commit)
    return b"".join(parts)


# ----------------------------
# Canonical value encoding
# ----------------------------

def encode_value(v: Any) -> bytes:
    if v is None:
        return b"N"
    if isinstance(v, bool):
        return b"B" + (b"\x01" if v else b"\x00")
    if isinstance(v, int):
        return b"I" + _i64_pack(v)
    if isinstance(v, bytes):
        return b"Y" + _u32(len(v)) + v
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"S" + _u32(len(b)) + b
    if isinstance(v, (list, tuple)):
        items = list(v)
        out = [b"T", _u32(len(items))]
        for it in items:
            b = encode_value(it)
            out.append(_u32(len(b)))
            out.append(b)
        return b"".join(out)
    if isinstance(v, dict):
        items = []
        for k in sorted(v.keys()):
            if not isinstance(k, str):
                raise TypeError("dict keys must be str for canonical encoding")
            items.append((k, v[k]))
        return encode_value(tuple(items))
    raise TypeError(f"unsupported type for encoding: {type(v)}")


def decode_value(b: bytes, offset: int = 0) -> Tuple[Any, int]:
    if offset >= len(b):
        raise ValueError("decode_value: empty")
    tag = b[offset : offset + 1]
    offset += 1
    if tag == b"N":
        return None, offset
    if tag == b"B":
        if offset >= len(b):
            raise ValueError("bad bool")
        v = b[offset] != 0
        return v, offset + 1
    if tag == b"I":
        if offset + 8 > len(b):
            raise ValueError("bad int")
        (n,) = struct.unpack("<q", b[offset : offset + 8])
        return n, offset + 8
    if tag == b"Y":
        if offset + 4 > len(b):
            raise ValueError("bad bytes len")
        (n,) = struct.unpack("<I", b[offset : offset + 4])
        offset += 4
        if offset + n > len(b):
            raise ValueError("bad bytes body")
        v = b[offset : offset + n]
        return v, offset + n
    if tag == b"S":
        if offset + 4 > len(b):
            raise ValueError("bad str len")
        (n,) = struct.unpack("<I", b[offset : offset + 4])
        offset += 4
        if offset + n > len(b):
            raise ValueError("bad str body")
        v = b[offset : offset + n].decode("utf-8")
        return v, offset + n
    if tag == b"T":
        if offset + 4 > len(b):
            raise ValueError("bad tuple len")
        (n,) = struct.unpack("<I", b[offset : offset + 4])
        offset += 4
        items = []
        for _ in range(n):
            if offset + 4 > len(b):
                raise ValueError("bad tuple item len")
            (ln,) = struct.unpack("<I", b[offset : offset + 4])
            offset += 4
            if offset + ln > len(b):
                raise ValueError("bad tuple item body")
            it, _ = decode_value(b[offset : offset + ln], 0)
            offset += ln
            items.append(it)
        return tuple(items), offset
    raise ValueError(f"unknown tag: {tag!r}")


# ----------------------------
# Core data model
# ----------------------------

@dataclass(frozen=True)
class Node:
    op: str
    parents: Tuple[Bytes32, ...]
    payload: bytes
    meta_commit: bytes
    h: Bytes32
    debug_id: int


class Store:
    def __init__(self) -> None:
        self.nodes: Dict[Bytes32, Node] = {}
        self._next_id: int = 1

    def add_node(
        self,
        op: str,
        parents: Sequence[Bytes32],
        payload: bytes = b"",
        meta_commit: bytes = b"",
    ) -> Bytes32:
        parents_t = tuple(parents)
        bts = canonical_node_bytes(op, parents_t, payload, meta_commit)
        h = sha256(bts)
        if h in self.nodes:
            return h
        n = Node(op=op, parents=parents_t, payload=payload, meta_commit=meta_commit, h=h, debug_id=self._next_id)
        self._next_id += 1
        self.nodes[h] = n
        return h

    def add_node_obj(
        self,
        op: str,
        parents: Sequence[Bytes32],
        payload_obj: Any = None,
        meta_commit: bytes = b"",
    ) -> Bytes32:
        payload = b"" if payload_obj is None else encode_value(payload_obj)
        return self.add_node(op, parents, payload=payload, meta_commit=meta_commit)

    def get(self, h: Bytes32) -> Node:
        try:
            return self.nodes[h]
        except KeyError:
            raise KeyError(f"missing node: {h.hex()}")

    def has(self, h: Bytes32) -> bool:
        return h in self.nodes

    def stats(self) -> Dict[str, Any]:
        payload_bytes = sum(len(n.payload) for n in self.nodes.values())
        meta_commit_bytes = sum(len(n.meta_commit) for n in self.nodes.values())
        return {
            "node_count": len(self.nodes),
            "payload_bytes": payload_bytes,
            "meta_commit_bytes": meta_commit_bytes,
        }

    def to_json(self) -> Dict[str, Any]:
        items = []
        for h, n in self.nodes.items():
            items.append(
                {
                    "h": h.hex(),
                    "op": n.op,
                    "parents": [p.hex() for p in n.parents],
                    "payload_b64": _b64e(n.payload),
                    "meta_commit_b64": _b64e(n.meta_commit),
                    "debug_id": n.debug_id,
                }
            )
        items.sort(key=lambda x: (x["h"]))
        return {"nodes": items}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Store":
        st = Store()
        st.nodes.clear()
        st._next_id = 1
        nodes = obj.get("nodes", [])
        for it in nodes:
            h = bytes.fromhex(it["h"])
            op = it["op"]
            parents = tuple(bytes.fromhex(p) for p in it["parents"])
            payload = _b64d(it["payload_b64"])
            meta_commit = _b64d(it["meta_commit_b64"])
            debug_id = int(it.get("debug_id", 0))
            bts = canonical_node_bytes(op, parents, payload, meta_commit)
            hh = sha256(bts)
            if hh != h:
                raise ValueError("store load: hash mismatch")
            st.nodes[h] = Node(op=op, parents=parents, payload=payload, meta_commit=meta_commit, h=h, debug_id=debug_id)
            st._next_id = max(st._next_id, debug_id + 1)
        return st


@dataclass
class State:
    signature_version: str
    roots: Dict[str, Bytes32]
    store: Store
    cache: Dict[Bytes32, Any]

    def __init__(self, signature_version: str, roots: Optional[Dict[str, Bytes32]] = None, store: Optional[Store] = None) -> None:
        self.signature_version = signature_version
        self.roots = dict(roots or {})
        self.store = store or Store()
        self.cache = {}

    def bind(self, name: str, h: Bytes32) -> None:
        self.roots[name] = h

    def unbind(self, name: str) -> None:
        if name in self.roots:
            del self.roots[name]

    def state_root(self) -> Bytes32:
        hs = [self.roots[k] for k in sorted(self.roots.keys())]
        buf = b"".join(hs)
        return sha256(b"ROOTS" + _u32(len(hs)) + buf)

    def to_json(self) -> Dict[str, Any]:
        return {
            "signature_version": self.signature_version,
            "roots": {k: v.hex() for k, v in sorted(self.roots.items())},
            "store": self.store.to_json(),
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "State":
        sig = obj.get("signature_version", "")
        if not isinstance(sig, str) or not sig:
            sig = "unknown"
        roots_in = obj.get("roots", {})
        roots: Dict[str, Bytes32] = {}
        for k, v in roots_in.items():
            roots[str(k)] = bytes.fromhex(v)
        st = Store.from_json(obj.get("store", {}))
        return State(signature_version=sig, roots=roots, store=st)


# ----------------------------
# Evaluator
# ----------------------------

class Evaluator:
    def __init__(self, state: State, strict_i64: bool, i64_wrap: bool, div_trunc0: bool) -> None:
        self.state = state
        self.strict_i64 = strict_i64
        self.i64_wrap = i64_wrap
        self.div_trunc0 = div_trunc0

    def _norm_i64(self, x: int) -> int:
        if self.i64_wrap:
            return _i64_wrap(x)
        if self.strict_i64:
            if x < -(1 << 63) or x > (1 << 63) - 1:
                raise OverflowError("i64 overflow")
        return x

    def eval(self, h: Bytes32) -> Any:
        if h in self.state.cache:
            return self.state.cache[h]
        n = self.state.store.get(h)
        op = n.op

        def ev(i: int) -> Any:
            return self.eval(n.parents[i])

        if op == "CONST":
            v, _ = decode_value(n.payload, 0)
            self.state.cache[h] = v
            return v

        if op == "NIL":
            v = ("NIL",)
            self.state.cache[h] = v
            return v

        if op == "CONS":
            head = ev(0)
            tail = ev(1)
            v = ("CONS", head, tail)
            self.state.cache[h] = v
            return v

        if op in ("ADD", "SUB", "MUL", "DIV"):
            a = ev(0)
            b = ev(1)
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(f"{op}: expected ints")
            if op == "ADD":
                r = a + b
            elif op == "SUB":
                r = a - b
            elif op == "MUL":
                r = a * b
            else:
                if b == 0:
                    raise ZeroDivisionError("DIV by zero")
                r = int(a / b) if self.div_trunc0 else (a // b)
            r = self._norm_i64(r)
            self.state.cache[h] = r
            return r

        if op == "CMP":
            a = ev(0)
            b = ev(1)
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError("CMP: expected ints")
            r = -1 if a < b else (1 if a > b else 0)
            self.state.cache[h] = r
            return r

        if op in ("EQ", "LT", "LE", "GT", "GE"):
            a = ev(0)
            b = ev(1)
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError(f"{op}: expected ints")
            if op == "EQ":
                r = 1 if a == b else 0
            elif op == "LT":
                r = 1 if a < b else 0
            elif op == "LE":
                r = 1 if a <= b else 0
            elif op == "GT":
                r = 1 if a > b else 0
            else:
                r = 1 if a >= b else 0
            self.state.cache[h] = r
            return r

        if op == "IF":
            c = ev(0)
            t = n.parents[1]
            f = n.parents[2]
            if not isinstance(c, int):
                raise TypeError("IF: expected int condition")
            out = self.eval(t if c != 0 else f)
            self.state.cache[h] = out
            return out

        if op == "TUPLE":
            items = [self.eval(p) for p in n.parents]
            v = tuple(items)
            self.state.cache[h] = v
            return v

        if op == "GET":
            tup = ev(0)
            idx = ev(1)
            if not isinstance(idx, int):
                raise TypeError("GET: index must be int")
            if not isinstance(tup, tuple):
                raise TypeError("GET: container must be tuple")
            v = tup[idx]
            self.state.cache[h] = v
            return v

        if op == "MERKLE":
            leaves = [p for p in n.parents]
            cur = leaves[:]
            if not cur:
                out = sha256(b"MERKLE0")
                self.state.cache[h] = out
                return out
            while len(cur) > 1:
                nxt = []
                for i in range(0, len(cur), 2):
                    a = cur[i]
                    b2 = cur[i + 1] if i + 1 < len(cur) else cur[i]
                    nxt.append(sha256(b"MK" + a + b2))
                cur = nxt
            out = cur[0]
            self.state.cache[h] = out
            return out

        if op == "ABSTRACT":
            desc, _ = decode_value(n.payload, 0)
            exemplars = [self.eval(p) for p in n.parents]
            v = ("ABSTRACT", desc, tuple(exemplars))
            self.state.cache[h] = v
            return v

        raise ValueError(f"unknown opcode: {op}")


# ----------------------------
# Constructors / helpers
# ----------------------------

def const_int(st: Store, n: int, meta_commit: bytes = b"") -> Bytes32:
    return st.add_node_obj("CONST", [], int(n), meta_commit=meta_commit)


def const_bytes(st: Store, b: bytes, meta_commit: bytes = b"") -> Bytes32:
    return st.add_node_obj("CONST", [], b, meta_commit=meta_commit)


def const_str(st: Store, s: str, meta_commit: bytes = b"") -> Bytes32:
    return st.add_node_obj("CONST", [], s, meta_commit=meta_commit)


def nil(st: Store) -> Bytes32:
    return st.add_node("NIL", [])


def cons(st: Store, head: Bytes32, tail: Bytes32) -> Bytes32:
    return st.add_node("CONS", [head, tail])


def list_(st: Store, items: List[Bytes32]) -> Bytes32:
    cur = nil(st)
    for it in reversed(items):
        cur = cons(st, it, cur)
    return cur


def add(st: Store, a: Bytes32, b: Bytes32) -> Bytes32:
    return st.add_node("ADD", [a, b])


def sub(st: Store, a: Bytes32, b: Bytes32) -> Bytes32:
    return st.add_node("SUB", [a, b])


def mul(st: Store, a: Bytes32, b: Bytes32) -> Bytes32:
    return st.add_node("MUL", [a, b])


def div(st: Store, a: Bytes32, b: Bytes32) -> Bytes32:
    return st.add_node("DIV", [a, b])


def eq(st: Store, a: Bytes32, b: Bytes32) -> Bytes32:
    return st.add_node("EQ", [a, b])


def merkle(st: Store, leaves: List[Bytes32]) -> Bytes32:
    return st.add_node("MERKLE", leaves)


# ----------------------------
# Graph closure + signatures
# ----------------------------

def trace_closure(store: Store, roots: Iterable[Bytes32], max_nodes: Optional[int] = None) -> Set[Bytes32]:
    seen: Set[Bytes32] = set()
    stack = list(roots)
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        if not store.has(h):
            continue
        seen.add(h)
        if max_nodes is not None and len(seen) > max_nodes:
            break
        n = store.get(h)
        for p in n.parents:
            if p not in seen:
                stack.append(p)
    return seen


def canon_signature(store: Store, h: Bytes32, max_depth: int = 8) -> str:
    def sig(node_h: Bytes32, d: int) -> str:
        if not store.has(node_h):
            return "MISSING(?)"
        n = store.get(node_h)
        op = n.op
        if d <= 0:
            return f"{op}(...)"
        if op == "CONST":
            try:
                v, _ = decode_value(n.payload, 0)
                if isinstance(v, int) and -(1 << 63) <= v <= (1 << 63) - 1 and abs(v) <= 8:
                    return f"CONST:{v}"
                if isinstance(v, str) and len(v) <= 12:
                    return f"CONST:S:{v}"
                if isinstance(v, bytes) and len(v) <= 8:
                    return f"CONST:Y:{v.hex()}"
                body = n.payload[1:]
                h8 = hashlib.sha256(body).hexdigest()[:8]
                t = n.payload[:1].decode("ascii", errors="replace")
                return f"CONST:{t}:h#{h8}"
            except Exception:
                return "CONST:?("
        if op == "ABSTRACT":
            try:
                desc, _ = decode_value(n.payload, 0)
                if isinstance(desc, str) and len(desc) <= 32:
                    return f"ABSTRACT:{desc}({','.join(sig(p, d-1) for p in n.parents)})"
            except Exception:
                pass
            return f"ABSTRACT({','.join(sig(p, d-1) for p in n.parents)})"
        return f"{op}({','.join(sig(p, d-1) for p in n.parents)})"
    return sig(h, max_depth)


def learn_pass(
    state: State,
    threshold: int,
    max_cluster: int,
    prefix: str,
    signature_depth: int,
) -> Dict[str, Any]:
    store = state.store
    live = trace_closure(store, state.roots.values())
    by_sig: Dict[str, List[Bytes32]] = {}
    for h in live:
        s = canon_signature(store, h, max_depth=signature_depth)
        by_sig.setdefault(s, []).append(h)

    clusters = []
    created = []
    abstract_created = 0

    for s, hs in by_sig.items():
        hs = list(dict.fromkeys(hs))
        if len(hs) < threshold:
            continue
        hs.sort(key=lambda x: x.hex())
        exemplars = hs[:max_cluster]
        desc = f"{prefix}:abstract:{hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]}"
        payload = encode_value(desc)
        ah = store.add_node("ABSTRACT", exemplars, payload=payload)
        base = desc
        idx = 0
        while True:
            nm = f"{base}:{idx}"
            if nm not in state.roots:
                break
            idx += 1
        state.bind(nm, ah)
        abstract_created += 1
        created.append({"name": nm, "hash": ah.hex(), "signature": s, "size": len(hs)})
        clusters.append({"signature": s, "count": len(hs), "exemplars": [x.hex() for x in exemplars], "abstract": nm})

    return {"clusters": len(by_sig), "abstract_created": abstract_created, "created": created, "cluster_detail": clusters}


# ----------------------------
# DOT export + rendering
# ----------------------------

def dot_export_from_hash(
    state: State,
    root_hash: Bytes32,
    out_path: str,
    max_nodes: int,
    max_depth: int,
    missing_ok: bool = True,
) -> Dict[str, Any]:
    """
    Robust DOT export:
      - If a referenced parent hash is missing from the store, we emit a stub node
        (so proofs / partial stores never crash).
    """
    store = state.store
    nodes_written = 0
    edges_written = 0
    visited: Set[Bytes32] = set()
    stubbed: int = 0

    def label_for_node(n: Node) -> str:
        if n.op == "CONST":
            try:
                v, _ = decode_value(n.payload, 0)
                if isinstance(v, int):
                    return f"CONST\\n{v}"
                if isinstance(v, str):
                    t = v if len(v) <= 24 else v[:21] + "..."
                    return f"CONST\\n{t}"
                if isinstance(v, bytes):
                    t = v.hex()
                    if len(t) > 24:
                        t = t[:21] + "..."
                    return f"CONST\\n0x{t}"
                return "CONST"
            except Exception:
                return "CONST"
        if n.op == "ABSTRACT":
            try:
                desc, _ = decode_value(n.payload, 0)
                if isinstance(desc, str):
                    t = desc if len(desc) <= 28 else desc[:25] + "..."
                    return f"ABSTRACT\\n{t}"
            except Exception:
                pass
            return "ABSTRACT"
        return n.op

    def color_for(op: str) -> str:
        if op == "CONST":
            return "lightyellow"
        if op in ("ADD", "SUB", "MUL", "DIV", "CMP", "EQ", "LT", "LE", "GT", "GE"):
            return "lightblue"
        if op in ("CONS", "NIL"):
            return "moccasin"
        if op == "ABSTRACT":
            return "lightgreen"
        if op in ("MERKLE",):
            return "plum"
        return "white"

    lines = [
        "digraph G {",
        "  rankdir=TB;",
        "  node [shape=box, style=filled, fontname=Helvetica];",
    ]

    def add_stub(h: Bytes32) -> None:
        nonlocal stubbed, nodes_written
        hid = h.hex()[:12]
        lines.append(f'  "{hid}" [label="MISSING\\n{hid}", fillcolor=gray90];')
        stubbed += 1
        nodes_written += 1

    stack: List[Tuple[Bytes32, int]] = [(root_hash, 0)]
    while stack and nodes_written < max_nodes:
        h, d = stack.pop()
        if h in visited:
            continue
        visited.add(h)

        hid = h.hex()[:12]

        if not store.has(h):
            if missing_ok:
                add_stub(h)
                continue
            raise KeyError(f"missing node: {h.hex()}")

        n = store.get(h)
        lab = label_for_node(n)
        color = color_for(n.op)
        lines.append(f'  "{hid}" [label="{lab}\\n{hid}", fillcolor={color}];')
        nodes_written += 1

        if d >= max_depth:
            continue

        for p in n.parents:
            pid = p.hex()[:12]
            lines.append(f'  "{hid}" -> "{pid}";')
            edges_written += 1
            if p not in visited:
                stack.append((p, d + 1))

    lines.append("}")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, out_path)
    return {
        "nodes_written": nodes_written,
        "edges_written": edges_written,
        "max_nodes": max_nodes,
        "max_depth": max_depth,
        "missing_ok": bool(missing_ok),
        "stubbed": stubbed,
    }


def render_dot(dot_path: str, out_path: str, fmt: str) -> None:
    dot = _which("dot")
    if not dot:
        raise RuntimeError("graphviz 'dot' not found. In Colab: apt-get install graphviz")
    tmp = out_path + ".tmp"
    subprocess.check_call([dot, f"-T{fmt}", dot_path, "-o", tmp])
    os.replace(tmp, out_path)


# ----------------------------
# Proof extraction
# ----------------------------

def proof_path_parents(store: Store, root: Bytes32, target: Bytes32) -> Optional[Dict[Bytes32, Bytes32]]:
    if not store.has(root) or not store.has(target):
        return None
    q = [root]
    prev: Dict[Bytes32, Optional[Bytes32]] = {root: None}
    while q:
        cur = q.pop(0)
        if cur == target:
            break
        if not store.has(cur):
            continue
        n = store.get(cur)
        for p in n.parents:
            if p not in prev:
                prev[p] = cur
                q.append(p)
    if target not in prev:
        return None
    out: Dict[Bytes32, Bytes32] = {}
    cur = target
    while True:
        parent = prev[cur]
        if parent is None:
            break
        out[cur] = parent
        cur = parent
    return out


def minimal_proof_nodes(store: Store, root: Bytes32, target: Bytes32) -> List[Node]:
    prev = proof_path_parents(store, root, target)
    if prev is None:
        raise ValueError("target not reachable from root (or missing nodes)")
    path_nodes: Set[Bytes32] = set(prev.keys()) | {root}
    proof_set: Set[Bytes32] = set()
    stack = [root]
    seen: Set[Bytes32] = set()
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        seen.add(h)
        if not store.has(h):
            continue
        proof_set.add(h)
        n = store.get(h)
        if h == target:
            continue
        for p in n.parents:
            if p in path_nodes:
                stack.append(p)
    nodes = [store.get(h) for h in sorted(proof_set, key=lambda x: x.hex())]
    return nodes


def proof_object(state: State, root: Bytes32, target: Bytes32) -> Dict[str, Any]:
    nodes = minimal_proof_nodes(state.store, root, target)
    return {
        "signature_version": state.signature_version,
        "root_hash": root.hex(),
        "target_hash": target.hex(),
        "nodes": [
            {
                "h": n.h.hex(),
                "op": n.op,
                "parents": [p.hex() for p in n.parents],
                "payload_b64": _b64e(n.payload),
                "meta_commit_b64": _b64e(n.meta_commit),
            }
            for n in nodes
        ],
    }


# ----------------------------
# GC prune
# ----------------------------

def gc_prune(state: State) -> Dict[str, Any]:
    store = state.store
    live = trace_closure(store, state.roots.values())
    before = len(store.nodes)
    new_nodes = {h: n for h, n in store.nodes.items() if h in live}
    after = len(new_nodes)
    store.nodes = new_nodes
    state.cache.clear()
    return {"before": before, "after": after, "collected": before - after}


# ----------------------------
# Demo builder
# ----------------------------

def build_demo_state(
    signature_version: str,
    strict_i64: bool,
    i64_wrap: bool,
    div_trunc0: bool,
    event_nonces: bool,
) -> Tuple[State, Dict[str, Any]]:
    state = State(signature_version=signature_version)
    st = state.store

    nonce_ctr = 1

    def uconst_int(n: int) -> Bytes32:
        nonlocal nonce_ctr
        mc = b""
        if event_nonces:
            mc = _u64(nonce_ctr)
            nonce_ctr += 1
        return const_int(st, n, meta_commit=mc)

    one_a = uconst_int(1)
    one_b = uconst_int(1)
    one_c = uconst_int(1)
    one_d = uconst_int(1)
    one_e = uconst_int(1)
    one_f = uconst_int(1)

    zero = uconst_int(0)

    two_1 = add(st, one_a, one_b)
    two_2 = add(st, one_c, one_d)
    two_3 = add(st, one_e, one_f)

    four_bal = add(st, two_1, two_2)
    four_chain = add(st, add(st, add(st, one_a, one_b), one_c), one_d)

    lst1 = list_(st, [one_a, one_b, one_c])
    lst2 = list_(st, [one_d, one_e, one_f])
    lst3 = list_(st, [uconst_int(1), uconst_int(1), uconst_int(1)])

    eq_4s = eq(st, four_bal, four_chain)
    assert_4s = eq(st, uconst_int(4), four_bal)

    root_commit = merkle(st, [four_bal, four_chain, lst1, lst2, lst3, eq_4s, assert_4s, two_1, two_2, two_3, zero])

    state.bind("two_1", two_1)
    state.bind("two_2", two_2)
    state.bind("two_3", two_3)
    state.bind("four_bal", four_bal)
    state.bind("four_chain", four_chain)
    state.bind("lst1", lst1)
    state.bind("lst2", lst2)
    state.bind("lst3", lst3)
    state.bind("eq_4s", eq_4s)
    state.bind("assert_4s", assert_4s)
    state.bind("root_commit", root_commit)

    ev = Evaluator(state, strict_i64=strict_i64, i64_wrap=i64_wrap, div_trunc0=div_trunc0)
    demo_eval: Dict[str, Any] = {}
    for k in sorted(state.roots.keys()):
        h = state.roots[k]
        try:
            v = ev.eval(h)
            if isinstance(v, bytes):
                demo_eval[k] = {"hash": h.hex(), "eval": {"type": "bytes", "len": len(v), "b64": _b64e(v)}}
            else:
                demo_eval[k] = {"hash": h.hex(), "eval": v}
        except Exception as e:
            demo_eval[k] = {"hash": h.hex(), "eval_error": str(e)}

    return state, demo_eval


# ----------------------------
# Visualization bundle (DOT + images + proofs)
# ----------------------------

def render_visualizations_bundle(
    out_dir: str,
    state: State,
    chosen_roots: List[str],
    dot_depths: List[int],
    dot_max_nodes_list: List[int],
    render_formats: List[str],
    dot_default_max_nodes: int,
    dot_default_max_depth: int,
    proof_results: Dict[str, Any],
    render_proofs: bool,
    missing_ok: bool = True,
) -> Dict[str, Any]:
    vis: Dict[str, Any] = {}
    fmts = [f.lower() for f in render_formats if f.lower() in ("png", "svg")]
    depths = dot_depths[:]
    max_nodes_list = dot_max_nodes_list[:]

    for rn in chosen_roots:
        if rn not in state.roots:
            continue
        root_hash = state.roots[rn]

        for d in depths:
            for mn in max_nodes_list:
                tag = f"{rn}__depth{d}__max{mn}"
                dot_path = os.path.join(out_dir, f"{_safe_name(tag)}.dot")
                info = dot_export_from_hash(state, root_hash, dot_path, max_nodes=mn, max_depth=d, missing_ok=missing_ok)
                item = {"dot": os.path.basename(dot_path), "meta": info}
                for fmt in fmts:
                    out_img = os.path.join(out_dir, f"{_safe_name(tag)}.{fmt}")
                    render_dot(dot_path, out_img, fmt)
                    item[fmt] = os.path.basename(out_img)
                vis[tag] = item

        tag = f"{rn}__full"
        dot_path = os.path.join(out_dir, f"{_safe_name(tag)}.dot")
        info = dot_export_from_hash(
            state, root_hash, dot_path,
            max_nodes=dot_default_max_nodes,
            max_depth=dot_default_max_depth,
            missing_ok=missing_ok,
        )
        item = {"dot": os.path.basename(dot_path), "meta": info}
        for fmt in fmts:
            out_img = os.path.join(out_dir, f"{_safe_name(tag)}.{fmt}")
            render_dot(dot_path, out_img, fmt)
            item[fmt] = os.path.basename(out_img)
        vis[tag] = item

    if render_proofs:
        for rn, pr in proof_results.items():
            proof_file = pr.get("proof_file")
            if not proof_file:
                continue
            proof_path = os.path.join(out_dir, proof_file)
            pobj = _read_json(proof_path)
            nodes = pobj.get("nodes", [])
            if not isinstance(nodes, list) or not nodes:
                continue

            # Build a store from proof nodes only
            pst = Store()
            pst.nodes.clear()
            pst._next_id = 1
            for it in nodes:
                h = bytes.fromhex(it["h"])
                op = it["op"]
                parents = tuple(bytes.fromhex(p) for p in it["parents"])
                payload = _b64d(it["payload_b64"])
                meta_commit = _b64d(it["meta_commit_b64"])
                bts = canonical_node_bytes(op, parents, payload, meta_commit)
                hh = sha256(bts)
                if hh != h:
                    raise ValueError("proof node hash mismatch")
                pst.nodes[h] = Node(op=op, parents=parents, payload=payload, meta_commit=meta_commit, h=h, debug_id=pst._next_id)
                pst._next_id += 1

            proot = bytes.fromhex(pobj["root_hash"])
            pstate = State(signature_version=state.signature_version, roots={f"proof_root:{rn}": proot}, store=pst)

            tag = f"proof__{rn}__to_{pobj['target_hash'][:12]}"
            dot_path = os.path.join(out_dir, f"{_safe_name(tag)}.dot")
            info = dot_export_from_hash(pstate, proot, dot_path, max_nodes=20000, max_depth=999999, missing_ok=True)
            item = {"dot": os.path.basename(dot_path), "meta": info, "proof": os.path.basename(proof_file)}
            for fmt in fmts:
                out_img = os.path.join(out_dir, f"{_safe_name(tag)}.{fmt}")
                render_dot(dot_path, out_img, fmt)
                item[fmt] = os.path.basename(out_img)
            vis[tag] = item

    return vis


# ----------------------------
# Charts (matplotlib) — run diagnostics
# ----------------------------

def _compute_depths(store: Store, roots: Iterable[Bytes32], max_nodes: int = 250000) -> Dict[Bytes32, int]:
    """
    Compute "depth" = max distance to a leaf for reachable nodes.
    Uses memoized DFS with cycle protection (should be a DAG, but we still guard).
    """
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    memo: Dict[Bytes32, int] = {}
    visiting: Set[Bytes32] = set()
    count_seen = 0

    def depth(h: Bytes32) -> int:
        nonlocal count_seen
        if h in memo:
            return memo[h]
        if not store.has(h):
            memo[h] = 0
            return 0
        if h in visiting:
            # Should not happen for a real DAG. Break cycles safely.
            memo[h] = 0
            return 0
        visiting.add(h)
        n = store.get(h)
        if not n.parents:
            d = 0
        else:
            d = 1 + max(depth(p) for p in n.parents)
        visiting.remove(h)
        memo[h] = d
        count_seen += 1
        if count_seen > max_nodes:
            return d
        return d

    for r in roots:
        depth(r)
        if count_seen > max_nodes:
            break
    return memo


def _make_charts(out_dir: str, state: State, learn_res: Dict[str, Any]) -> Dict[str, str]:
    """
    Writes PNG charts to out_dir. Returns dict of chart keys -> filename.
    """
    # Import inside so the kernel still runs without matplotlib
    import matplotlib.pyplot as plt  # type: ignore

    charts: Dict[str, str] = {}

    # 1) Opcode histogram
    op_counts: Dict[str, int] = {}
    for n in state.store.nodes.values():
        op_counts[n.op] = op_counts.get(n.op, 0) + 1
    ops = sorted(op_counts.keys())
    vals = [op_counts[o] for o in ops]
    plt.figure()
    plt.bar(ops, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Opcode histogram")
    plt.tight_layout()
    fn = "chart_op_hist.png"
    plt.savefig(os.path.join(out_dir, fn), dpi=160)
    plt.close()
    charts["op_hist_png"] = fn

    # 2) Root closure sizes
    root_sizes: List[Tuple[str, int]] = []
    for k in sorted(state.roots.keys()):
        h = state.roots[k]
        live = trace_closure(state.store, [h], max_nodes=250000)
        root_sizes.append((k, len(live)))
    plt.figure()
    plt.bar([a for a, _ in root_sizes], [b for _, b in root_sizes])
    plt.xticks(rotation=45, ha="right")
    plt.title("Reachable nodes per root")
    plt.tight_layout()
    fn = "chart_root_sizes.png"
    plt.savefig(os.path.join(out_dir, fn), dpi=160)
    plt.close()
    charts["root_sizes_png"] = fn

    # 3) Depth histogram (over all reachable nodes from all roots)
    depths = _compute_depths(state.store, state.roots.values(), max_nodes=250000)
    dvals = list(depths.values())
    plt.figure()
    if dvals:
        plt.hist(dvals, bins=min(50, max(10, len(set(dvals)))))
    plt.title("Depth histogram (reachable nodes)")
    plt.tight_layout()
    fn = "chart_depth_hist.png"
    plt.savefig(os.path.join(out_dir, fn), dpi=160)
    plt.close()
    charts["depth_hist_png"] = fn

    # 4) Learn cluster sizes (for ABSTRACT created)
    created = learn_res.get("created", [])
    sizes = [int(it.get("size", 0)) for it in created if isinstance(it, dict)]
    plt.figure()
    if sizes:
        plt.hist(sizes, bins=min(20, max(5, len(set(sizes)))))
    plt.title("Learn clusters: sizes of created ABSTRACT groups")
    plt.tight_layout()
    fn = "chart_learn_clusters.png"
    plt.savefig(os.path.join(out_dir, fn), dpi=160)
    plt.close()
    charts["learn_clusters_png"] = fn

    return charts


# ----------------------------
# HTML Index
# ----------------------------

def write_html_index(out_dir: str, summary: Dict[str, Any]) -> str:
    def esc(s: str) -> str:
        import html
        return html.escape(s, quote=True)

    idx_path = os.path.join(out_dir, "INDEX.html")
    vis = summary.get("visualizations", {}) or {}
    charts = summary.get("charts", {}) or {}
    files = summary.get("files", {}) or {}

    vis_rows = []
    for k in sorted(vis.keys()):
        item = vis[k]
        title = esc(k)
        parts = [f"<h3>{title}</h3>"]
        if "dot" in item:
            parts.append(f"<div><a href='{esc(item['dot'])}'>DOT</a></div>")
        if "proof" in item:
            parts.append(f"<div><a href='{esc(item['proof'])}'>Proof JSON</a></div>")
        for fmt in ("png", "svg"):
            if fmt in item:
                parts.append(f"<div><a href='{esc(item[fmt])}'>{fmt.upper()}</a></div>")
                if fmt == "png":
                    parts.append(f"<div><img src='{esc(item[fmt])}' style='max-width:100%;border:1px solid #ddd;'/></div>")
        vis_rows.append("\n".join(parts))

    chart_rows = []
    for k in sorted(charts.keys()):
        fn = charts[k]
        chart_rows.append(
            f"<h3>{esc(k)}</h3>"
            f"<div><a href='{esc(fn)}'>{esc(fn)}</a></div>"
            f"<div><img src='{esc(fn)}' style='max-width:100%;border:1px solid #ddd;'/></div>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>MA Kernel Run Index</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 4px; }}
small {{ color: #666; }}
hr {{ margin: 24px 0; }}
h3 {{ margin-top: 28px; }}
code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>MA Kernel Run</h1>
<small>Directory: <code>{esc(os.path.basename(out_dir))}</code></small><br/>
<small>Signature: <code>{esc(str(summary.get("signature_version","")))}</code> · Run: <code>{esc(str(summary.get("run_id","")))}</code></small>
<hr/>
<h2>Core Files</h2>
<div><a href="{esc(files.get("run_summary","RUN_SUMMARY.json"))}">RUN_SUMMARY.json</a></div>
<div><a href="{esc(files.get("demo_state",""))}">state_demo.json</a></div>
<div><a href="{esc(files.get("learn_state",""))}">state_learn.json</a></div>
<div><a href="{esc(files.get("pruned_state",""))}">state_pruned.json</a></div>
<div><a href="{esc(files.get("dot_index",""))}">dot_index.json</a></div>
<div><a href="{esc(files.get("proof_index",""))}">proof_index.json</a></div>
<hr/>
<h2>Charts</h2>
{''.join(chart_rows) if chart_rows else "<div><em>No charts generated.</em></div>"}
<hr/>
<h2>Graphs / Proof Visualizations</h2>
{''.join(vis_rows) if vis_rows else "<div><em>No visualizations generated.</em></div>"}
</body>
</html>
"""
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html_doc)
    os.replace(tmp, idx_path)
    return idx_path


# ----------------------------
# Zip + Colab download
# ----------------------------

def zip_dir(dir_path: str, zip_path: str) -> str:
    import zipfile
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, dir_path)
                z.write(p, arcname=rel)
    os.replace(tmp, zip_path)
    return zip_path


def colab_download(path: str) -> bool:
    try:
        from google.colab import files  # type: ignore
        files.download(path)
        return True
    except Exception:
        return False


# ----------------------------
# Commands
# ----------------------------

def cmd_new(args: argparse.Namespace) -> None:
    st = State(signature_version=args.signature_version)
    _write_json(args.out, st.to_json())
    print(f"wrote: {args.out}")


def cmd_bind(args: argparse.Namespace) -> None:
    obj = _read_json(args.state)
    st = State.from_json(obj)
    st.signature_version = args.signature_version or st.signature_version
    name = args.name
    if args.int is not None:
        h = const_int(st.store, int(args.int))
    elif args.str is not None:
        h = const_str(st.store, str(args.str))
    elif args.bytes_b64 is not None:
        h = const_bytes(st.store, _b64d(args.bytes_b64))
    else:
        raise ValueError("bind: need --int or --str or --bytes-b64")
    st.bind(name, h)
    _write_json(args.out, st.to_json())
    print(f"bound {name} -> {h.hex()}")
    print(f"wrote: {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    if args.name:
        h = st.roots[args.name]
    else:
        h = bytes.fromhex(args.hash)
    ev = Evaluator(st, strict_i64=args.strict_i64, i64_wrap=args.i64_wrap, div_trunc0=args.div_trunc0)
    v = ev.eval(h)
    if isinstance(v, bytes):
        print(json.dumps({"type": "bytes", "len": len(v), "b64": _b64e(v)}, indent=2))
    else:
        print(v)


def cmd_stats(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    print("=== STATE ===")
    print(f"signature_version(snapshot): {st.signature_version}")
    print(f"state_root: {st.state_root().hex()}")
    print(f"roots: {len(st.roots)}")
    for k in sorted(st.roots.keys()):
        print(f"  {k}: {st.roots[k].hex()}")
    print("=== STORE ===")
    s = st.store.stats()
    s["cache_entries"] = len(st.cache)
    print(json.dumps(s, indent=2))


def cmd_learn(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    res = learn_pass(
        st,
        threshold=args.threshold,
        max_cluster=args.max_cluster,
        prefix=args.prefix,
        signature_depth=args.signature_depth,
    )
    _write_json(args.out, st.to_json())
    print(json.dumps({k: res[k] for k in ("clusters", "abstract_created", "created")}, indent=2))
    print(f"wrote: {args.out}")


def cmd_export_dot(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    if args.name:
        root_hash = st.roots[args.name]
        root_name = args.name
    else:
        root_hash = bytes.fromhex(args.hash)
        root_name = args.hash[:12]
    info = dot_export_from_hash(st, root_hash, args.out, max_nodes=args.max_nodes, max_depth=args.max_depth, missing_ok=args.missing_ok)
    payload = {
        "signature_version": st.signature_version,
        "root_name": root_name,
        "root_hash": root_hash.hex(),
        "out": args.out,
        **info,
    }
    print(json.dumps(payload, indent=2))


def cmd_proof(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    root = st.roots[args.root] if args.root in st.roots else bytes.fromhex(args.root)
    target_prefix = args.target.lower()
    matches = [h for h in st.store.nodes.keys() if h.hex().startswith(target_prefix)]
    if not matches:
        raise SystemExit(f"target prefix not found: {args.target}")
    if len(matches) > 1:
        raise SystemExit(f"target prefix ambiguous ({len(matches)} matches). Use longer prefix.")
    target = matches[0]
    pobj = proof_object(st, root, target)
    _write_json(args.out, pobj)
    print(f"wrote: {args.out}")


def cmd_gc(args: argparse.Namespace) -> None:
    st = State.from_json(_read_json(args.state))
    st.signature_version = args.signature_version or st.signature_version
    pre_root = st.state_root().hex()
    info = gc_prune(st)
    post_root = st.state_root().hex()
    _write_json(args.out, st.to_json())
    print(json.dumps({"state_root_pre": pre_root, "state_root_post": post_root, "gc_info": info}, indent=2))
    print(f"wrote: {args.out}")


def cmd_build_demo(args: argparse.Namespace) -> None:
    st, demo_eval = build_demo_state(
        signature_version=args.signature_version,
        strict_i64=args.strict_i64,
        i64_wrap=args.i64_wrap,
        div_trunc0=args.div_trunc0,
        event_nonces=args.event_nonces,
    )
    _write_json(args.out, st.to_json())
    if args.summary_out:
        _write_json(
            args.summary_out,
            {
                "demo_eval": demo_eval,
                "state_root": st.state_root().hex(),
                "roots": {k: v.hex() for k, v in st.roots.items()},
            },
        )
    print(f"wrote: {args.out}")


def cmd_auto(args: argparse.Namespace) -> None:
    tracemalloc.start()

    run_id = args.run_id or _now_run_id()
    out_dir = args.out_dir or f"ma_kernel_run_{run_id}"
    _ensure_dir(out_dir)

    summary: Dict[str, Any] = {
        "signature_version": args.signature_version,
        "run_id": run_id,
        "out_dir": out_dir,
        "flags": {
            "strict_i64": args.strict_i64,
            "i64_wrap": args.i64_wrap,
            "div_trunc0": args.div_trunc0,
            "event_nonces": args.event_nonces,
            "full_snapshot": args.full_snapshot,
            "learn_threshold": args.learn_threshold,
            "learn_max_cluster": args.learn_max_cluster,
            "learn_prefix": args.learn_prefix,
            "learn_signature_depth": args.learn_signature_depth,
            "dot_roots": args.dot_roots,
            "dot_max_nodes": args.dot_max_nodes,
            "dot_max_depth": args.dot_max_depth,
            "render": args.render,
            "render_proofs": args.render_proofs,
            "render_formats": args.render_formats,
            "render_depths": args.render_depths,
            "render_max_nodes": args.render_max_nodes,
            "missing_ok": args.missing_ok,
            "make_charts": args.make_charts,
        },
        "files": {},
    }

    # (1) Demo state
    demo_state, demo_eval = build_demo_state(
        signature_version=args.signature_version,
        strict_i64=args.strict_i64,
        i64_wrap=args.i64_wrap,
        div_trunc0=args.div_trunc0,
        event_nonces=args.event_nonces,
    )
    demo_path = os.path.join(out_dir, args.demo_state)
    _write_json(demo_path, demo_state.to_json())
    summary["files"]["demo_state"] = args.demo_state
    summary["demo_eval"] = demo_eval

    # (2) Learn pass
    learn_state = State.from_json(demo_state.to_json())
    learn_res_full = learn_pass(
        learn_state,
        threshold=args.learn_threshold,
        max_cluster=args.learn_max_cluster,
        prefix=args.learn_prefix,
        signature_depth=args.learn_signature_depth,
    )
    learn_path = os.path.join(out_dir, args.learn_state)
    _write_json(learn_path, learn_state.to_json())
    summary["files"]["learn_state"] = args.learn_state
    summary["learn"] = {k: learn_res_full[k] for k in ("clusters", "abstract_created", "created")}
    summary["chosen_roots"] = []

    # Choose roots for DOT/proof pipeline
    chosen_roots: List[str] = []
    for r in args.dot_roots:
        if r in learn_state.roots:
            chosen_roots.append(r)
    if not chosen_roots:
        chosen_roots = sorted(list(learn_state.roots.keys()))[:3]
    summary["chosen_roots"] = chosen_roots

    # (3) DOT index
    dot_index: Dict[str, Any] = {}
    for rn in chosen_roots:
        dot_path = os.path.join(out_dir, f"{_safe_name(rn)}.dot")
        info = dot_export_from_hash(
            learn_state,
            learn_state.roots[rn],
            dot_path,
            max_nodes=args.dot_max_nodes,
            max_depth=args.dot_max_depth,
            missing_ok=args.missing_ok,
        )
        dot_index[rn] = {"dot": os.path.basename(dot_path), "root_hash": learn_state.roots[rn].hex(), **info}
    dot_index_path = os.path.join(out_dir, args.dot_index)
    _write_json(dot_index_path, dot_index)
    summary["files"]["dot_index"] = args.dot_index

    # (4) Proofs
    proof_index: Dict[str, Any] = {}
    proof_results: Dict[str, Any] = {}
    for rn in chosen_roots:
        root_hash = learn_state.roots[rn]
        # Pick a target: first parent if present, else the root itself.
        if learn_state.store.has(root_hash):
            n = learn_state.store.get(root_hash)
            target = root_hash if not n.parents else n.parents[0]
        else:
            target = root_hash
        proof_file = f"proof_{_safe_name(rn)}.json"
        proof_path = os.path.join(out_dir, proof_file)
        pobj = proof_object(learn_state, root_hash, target)
        _write_json(proof_path, pobj)
        proof_index[rn] = {
            "proof": proof_file,
            "root_hash": root_hash.hex(),
            "target_hash": target.hex(),
            "node_count": len(pobj["nodes"]),
        }
        proof_results[rn] = {"proof_file": proof_file, "root_hash": root_hash.hex(), "target_hash": target.hex()}

    proof_index_path = os.path.join(out_dir, args.proof_index)
    _write_json(proof_index_path, proof_index)
    summary["files"]["proof_index"] = args.proof_index

    # (5) GC prune snapshot
    pruned_state = State.from_json(learn_state.to_json())
    pre_root = pruned_state.state_root().hex()
    store_pre = pruned_state.store.stats()
    store_pre["cache_entries"] = len(pruned_state.cache)
    gc_info = gc_prune(pruned_state)
    post_root = pruned_state.state_root().hex()
    store_post = pruned_state.store.stats()
    store_post["cache_entries"] = len(pruned_state.cache)
    pruned_path = os.path.join(out_dir, args.pruned_state)
    _write_json(pruned_path, pruned_state.to_json())
    summary["files"]["pruned_state"] = args.pruned_state
    summary["gc"] = {
        "state_root_pre": pre_root,
        "state_root_post": post_root,
        "gc_info": gc_info,
        "store_pre": store_pre,
        "store_post": store_post,
    }

    # (6) Environment + memory stats
    cur, peak = tracemalloc.get_traced_memory()
    summary["memory"] = {
        "tracemalloc_current_mb": round(cur / (1024 * 1024), 6),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 6),
    }
    summary["env"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "colab": _is_colab(),
        "graphviz_dot": bool(_which("dot")),
        "ipython": _in_ipython(),
    }

    # (7) Render visualizations
    if args.render:
        try:
            vis = render_visualizations_bundle(
                out_dir=out_dir,
                state=learn_state,
                chosen_roots=chosen_roots,
                dot_depths=args.render_depths,
                dot_max_nodes_list=args.render_max_nodes,
                render_formats=args.render_formats,
                dot_default_max_nodes=args.dot_max_nodes,
                dot_default_max_depth=args.dot_max_depth,
                proof_results=proof_results,
                render_proofs=args.render_proofs,
                missing_ok=args.missing_ok,
            )
            summary["visualizations"] = vis
        except Exception as e:
            summary["visualization_error"] = str(e)

    # (8) Charts
    if args.make_charts:
        try:
            charts = _make_charts(out_dir, learn_state, learn_res_full)
            summary["charts"] = charts
        except Exception as e:
            summary["charts_error"] = str(e)

    # (9) Write HTML index last (so it can reference whatever exists)
    try:
        idx_path = write_html_index(out_dir, summary)
        summary["files"]["html_index"] = os.path.basename(idx_path)
    except Exception as e:
        summary["html_index_error"] = str(e)

    # (10) Write run summary JSON
    run_summary_path = os.path.join(out_dir, "RUN_SUMMARY.json")
    summary["files"]["run_summary"] = "RUN_SUMMARY.json"
    _write_json(run_summary_path, summary)

    # (11) Zip
    zip_name = args.zip_name or f"{out_dir}.zip"
    zip_path = os.path.join(os.getcwd(), zip_name)
    zip_dir(out_dir, zip_path)

    # Print human + machine outputs
    print(json.dumps(summary, indent=2))
    print()
    print(f"WROTE_DIR: {out_dir}")
    print(f"WROTE_ZIP: {zip_name}")

    # (12) Colab download
    if args.colab_download:
        ok = colab_download(zip_name)
        if not ok:
            print("NOTE: colab download not available in this environment.")


# ----------------------------
# Argparse
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ma_kernel.py")
    ap.add_argument("--signature-version", default="2026-01-18.sig.v1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("--out", required=True)
    p_new.set_defaults(fn=cmd_new)

    p_bind = sub.add_parser("bind")
    p_bind.add_argument("--state", required=True)
    p_bind.add_argument("--out", required=True)
    p_bind.add_argument("--name", required=True)
    g = p_bind.add_mutually_exclusive_group(required=True)
    g.add_argument("--int", type=int)
    g.add_argument("--str", type=str)
    g.add_argument("--bytes-b64", type=str)
    p_bind.set_defaults(fn=cmd_bind)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--state", required=True)
    p_eval.add_argument("--name", type=str, default=None)
    p_eval.add_argument("--hash", type=str, default=None)
    p_eval.add_argument("--strict-i64", action="store_true", default=False)
    p_eval.add_argument("--i64-wrap", action="store_true", default=False)
    p_eval.add_argument("--div-trunc0", action="store_true", default=False)
    p_eval.set_defaults(fn=cmd_eval)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--state", required=True)
    p_stats.set_defaults(fn=cmd_stats)

    p_learn = sub.add_parser("learn")
    p_learn.add_argument("--state", required=True)
    p_learn.add_argument("--out", required=True)
    p_learn.add_argument("--threshold", type=int, default=3)
    p_learn.add_argument("--max-cluster", type=int, default=12)
    p_learn.add_argument("--prefix", type=str, default="learn")
    p_learn.add_argument("--signature-depth", type=int, default=8)
    p_learn.set_defaults(fn=cmd_learn)

    p_dot = sub.add_parser("export-dot")
    p_dot.add_argument("--state", required=True)
    p_dot.add_argument("--name", type=str, default=None)
    p_dot.add_argument("--hash", type=str, default=None)
    p_dot.add_argument("--out", required=True)
    p_dot.add_argument("--max-nodes", type=int, default=5000)
    p_dot.add_argument("--max-depth", type=int, default=64)
    p_dot.add_argument("--missing-ok", action="store_true", default=True)
    p_dot.add_argument("--no-missing-ok", action="store_false", dest="missing_ok")
    p_dot.set_defaults(fn=cmd_export_dot)

    p_proof = sub.add_parser("proof")
    p_proof.add_argument("--state", required=True)
    p_proof.add_argument("--root", required=True, help="root name in state roots OR hex hash")
    p_proof.add_argument("--target", required=True, help="target hash prefix (hex)")
    p_proof.add_argument("--out", required=True)
    p_proof.set_defaults(fn=cmd_proof)

    p_gc = sub.add_parser("gc")
    p_gc.add_argument("--state", required=True)
    p_gc.add_argument("--out", required=True)
    p_gc.set_defaults(fn=cmd_gc)

    p_demo = sub.add_parser("build-demo")
    p_demo.add_argument("--out", required=True)
    p_demo.add_argument("--summary-out", type=str, default=None)
    p_demo.add_argument("--strict-i64", action="store_true", default=False)
    p_demo.add_argument("--i64-wrap", action="store_true", default=False)
    p_demo.add_argument("--div-trunc0", action="store_true", default=False)
    p_demo.add_argument("--event-nonces", action="store_true", default=True)
    p_demo.add_argument("--no-event-nonces", action="store_false", dest="event_nonces")
    p_demo.set_defaults(fn=cmd_build_demo)

    p_auto = sub.add_parser("auto")
    p_auto.add_argument("--run-id", type=str, default=None)
    p_auto.add_argument("--out-dir", type=str, default=None)
    p_auto.add_argument("--strict-i64", action="store_true", default=False)
    p_auto.add_argument("--i64-wrap", action="store_true", default=False)
    p_auto.add_argument("--div-trunc0", action="store_true", default=False)
    p_auto.add_argument("--event-nonces", action="store_true", default=True)
    p_auto.add_argument("--no-event-nonces", action="store_false", dest="event_nonces")
    p_auto.add_argument("--full-snapshot", action="store_true", default=False)
    p_auto.add_argument("--learn-threshold", type=int, default=3)
    p_auto.add_argument("--learn-max-cluster", type=int, default=12)
    p_auto.add_argument("--learn-prefix", type=str, default="learn")
    p_auto.add_argument("--learn-signature-depth", type=int, default=8)
    p_auto.add_argument("--demo-state", type=str, default="state_demo.json")
    p_auto.add_argument("--learn-state", type=str, default="state_learn.json")
    p_auto.add_argument("--dot-index", type=str, default="dot_index.json")
    p_auto.add_argument("--proof-index", type=str, default="proof_index.json")
    p_auto.add_argument("--pruned-state", type=str, default="state_pruned.json")
    p_auto.add_argument("--dot-roots", type=str, nargs="*", default=["four_bal", "four_chain", "lst1"])
    p_auto.add_argument("--dot-max-nodes", type=int, default=5000)
    p_auto.add_argument("--dot-max-depth", type=int, default=64)

    p_auto.add_argument("--render", action="store_true", default=True)
    p_auto.add_argument("--no-render", action="store_false", dest="render")
    p_auto.add_argument("--render-proofs", action="store_true", default=True)
    p_auto.add_argument("--no-render-proofs", action="store_false", dest="render_proofs")
    p_auto.add_argument("--render-formats", type=str, nargs="*", default=["png", "svg"])
    p_auto.add_argument("--render-depths", type=int, nargs="*", default=[6, 10, 16, 24, 40, 64])
    p_auto.add_argument("--render-max-nodes", type=int, nargs="*", default=[300, 800, 2000, 5000])

    p_auto.add_argument("--missing-ok", action="store_true", default=True)
    p_auto.add_argument("--no-missing-ok", action="store_false", dest="missing_ok")

    p_auto.add_argument("--make-charts", action="store_true", default=True)
    p_auto.add_argument("--no-make-charts", action="store_false", dest="make_charts")

    p_auto.add_argument("--zip-name", type=str, default=None)
    p_auto.add_argument("--colab-download", action="store_true", default=False)
    p_auto.set_defaults(fn=cmd_auto)

    return ap


# ----------------------------
# Jupyter/Colab argv normalization
# ----------------------------

def _normalize_argv(argv: Optional[List[str]]) -> List[str]:
    """
    Fixes the classic notebook issue where the kernel passes a path like:
      /root/.local/share/jupyter/runtime/kernel-....json
    and argparse thinks it's the command.
    """
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # No args => default to auto
    if not argv:
        return ["auto"]

    # Drop notebook kernel JSON path if present at head
    if argv:
        a0 = argv[0]
        base = os.path.basename(a0)
        if a0.endswith(".json") and ("kernel-" in base or "jupyter" in a0 or "runtime" in a0):
            argv = argv[1:]

    if not argv:
        return ["auto"]

    cmds = {"new", "bind", "eval", "stats", "learn", "export-dot", "proof", "gc", "build-demo", "auto"}

    # If the first remaining token is not a command, assume they intended auto
    if argv[0] not in cmds:
        # Some environments pass a config path as first arg; drop if it looks like a file
        if argv[0].endswith(".json") and os.path.exists(argv[0]):
            argv = argv[1:]
        if not argv:
            return ["auto"]
        if argv[0] not in cmds:
            return ["auto"] + argv

    return argv


def main(argv: Optional[List[str]] = None) -> None:
    argv2 = _normalize_argv(argv)
    ap = build_argparser()
    args = ap.parse_args(argv2)

    # eval: exactly one of --name/--hash
    if args.cmd == "eval":
        if (args.name is None) == (args.hash is None):
            raise SystemExit("eval: provide exactly one of --name or --hash")

    # export-dot: exactly one of --name/--hash
    if args.cmd == "export-dot":
        if (args.name is None) == (args.hash is None):
            raise SystemExit("export-dot: provide exactly one of --name or --hash")

    args.signature_version = getattr(args, "signature_version", None) or "2026-01-18.sig.v1"
    args.fn(args)


if __name__ == "__main__":
    main()
