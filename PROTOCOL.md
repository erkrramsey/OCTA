# NUMCHAIN × MAΔ  
## Full Protocol Specification & System Description (v1.4.x – Closed Alpha)

**Status:** Closed-alpha, specification-complete  
**Audience:** Implementers, researchers, auditors, contributors  
**Scope:** Trusted validator clusters (≈3–10 nodes), LAN / localhost  
**Guarantee:** Bit-exact deterministic replay across platforms  

---

## 0. What This Is

NUMCHAIN × MAΔ is a **deterministic, sharded, distributed memory system**.

It is **not** a general blockchain VM.

It is a purpose-built substrate for **verifiable, bounded, long-lived memory evolution**, where:

- State = structured tensors (not accounts, balances, or bytecode)
- Transactions = **memory deltas**, not programs
- Consensus exists solely to **order and replicate memory changes**
- Replay from genesis must be **bit-identical forever**

This system exists to explore **distributed cognitive memory**, multi-agent knowledge accumulation, and contradiction-aware state evolution under strong auditability constraints.

---

## 1. High-Level Architecture

┌──────────────┐ │  Producers   │  (AI agents, robots, tools) │ (MA logic)   │ └──────┬───────┘ │ MADelta (binary) ▼ ┌───────────────────────────┐ │        NUMCHAIN           │ │  ──────────────────────  │ │  • RPC / TX Ingest        │ │  • Gossip (TX + Consensus)│ │  • Leader-based BFT       │ │  • Deterministic Replay   │ │  • Append-only Log        │ └──────┬────────────────────┘ │ Ordered MADeltas ▼ ┌───────────────────────────┐ │  MA State (per shard)     │ │  • Topics                │ │  • POS / NEG / RES banks │ │  • Prototypes             │ │  • Bounded tensor dynamics│ └───────────────────────────┘

### Separation of Concerns

| Layer | Responsibility |
|-----|----------------|
| Producer | Semantic interpretation, similarity, novelty, contradiction |
| MADelta | Canonical, compact mutation description |
| NUMCHAIN | Ordering, replication, replay, safety |
| MA Apply | Pure integer state transition |

This separation is **intentional** and enforced.

---

## 2. Explicit Non-Goals (v1)

Out of scope by design:

- General smart contracts / VMs
- Floating-point arithmetic anywhere in consensus or apply
- Dynamic validator sets
- Permissionless adversarial networking
- WAN hardening (libp2p, NAT traversal)
- Fee markets, gas, MEV
- Zero-knowledge or privacy layers
- Orthospace multi-head projections (reserved)
- Sparse vector codecs (reserved)

---

## 3. Determinism & Replay Contract

**Invariant:**  
Given the same genesis and the same ordered MADeltas, all correct nodes **must** reach the same state bit-for-bit.

To enforce this:

- All arithmetic is integer-only
- All binary formats are fixed-layout
- All tie-breaks are explicitly specified
- All randomness is forbidden or genesis-seeded
- All rounding rules are normative

Any implementation violating these rules is **non-compliant**.

---

## 4. Fixed-Point Arithmetic (Locked)

### 4.1 Vector Representation

- Payload vectors: `int16`, Q8.8
- Internal vectors: `int32`, Q8.8
- Promotion rule: **sign-extend only**

int32 = (int32)int16

No shifting. No scaling. No reinterpretation.

### 4.2 Learning Rates

- `u16`, Q0.16
- `0` = use genesis default

### 4.3 Confidence

- `u16`, Q4.12
- Saturating addition only

---

## 5. MADelta Binary Format (v1)

### 5.1 Header (64 bytes, fixed)

| Offset | Type | Field |
|------:|------|------|
| 0 | u32 | magic = `0x4D414431` (`MAD1`) |
| 4 | u16 | version = `1` |
| 6 | u16 | flags |
| 8 | u64 | chain_id_hash |
| 16 | u64 | height_hint |
| 24 | u64 | timestamp_ns |
| 32 | u32 | topic_id |
| 36 | u64 | topic_tag |
| 44 | u8 | bank |
| 45 | u8 | action |
| 46 | u8 | vec_codec |
| 47 | u8 | proto_idx |
| 48 | u16 | dim |
| 50 | u16 | reserved (0) |
| 52 | u16 | delta_conf_q |
| 54 | u16 | lr_q |
| 56 | u32 | payload_len |
| 60 | — | padding |

### 5.2 Enums

**Bank**
- 1 = POS
- 2 = NEG
- 3 = RES (reserved)

**Action**
- 0 = EMA_ADD
- 1 = REPLACE_ABS
- 2 = DRIFT_ADD
- 3 = CONF_ONLY

**Vector Codec**
- 0 = Dense int16 Q8.8 (only valid v1)

### 5.3 Payload

- If action ≠ CONF_ONLY: `dim × int16`
- If CONF_ONLY: payload_len must be 0

### 5.4 Provenance (Optional)

If `HAS_PROVENANCE` flag set:

- Append 32 bytes (SHA-256 hash)
- Never interpreted by consensus logic

### 5.5 Size (dim = 256)

- Without provenance: **576 bytes**
- With provenance: **608 bytes**

---

## 6. Genesis Configuration (Immutable)

```toml
dim                = 256
K_pos              = 6
K_neg              = 3

per_dim_clip_q     = 5120    # ≈ 20.0
max_norm_q         = 2560    # ≈ 10.0

default_lr_q       = 6554    # ≈ 0.1
gentle_lr_q        = 655     # ≈ 0.01

plastic_th_sim_q15 = 14746   # ≈ 0.45
novel_th_sim_q15   = 11469   # ≈ 0.35


---

7. Deterministic Apply Rules (Kernel Side)

7.1 Learning Rate Selection

EMA_ADD: override if non-zero else default

DRIFT_ADD: use gentle_lr_q

REPLACE_ABS: no lr

CONF_ONLY: no vector change


7.2 Vector Mutation

EMA / DRIFT

ds = (payload_i16 × lr_q) >> 16
v = v + ds

REPLACE

v = sign_extend(payload)

7.3 Mandatory Per-Dimension Clip

Applied before L2 clamp:

v[i] = clamp(v[i], -per_dim_clip_q, +per_dim_clip_q)

7.4 Deterministic L2 Clamp (No Normalization)

Enforce ||v||₂ ≤ max_norm_q

Integer-only

Zero vector preserved

Normative isqrt_u64 implementation


(see prior section for exact code — mandatory)

7.5 Confidence & Usage

conf += delta_conf_q (saturating)

usage += 1

On REPLACE: usage = max(1, usage / 2)



---

8. Producer Rules (Semantic Side)

Producers are not free-form.
They must follow these rules to ensure semantic convergence.

8.1 Float → Fixed Conversion

Q8.8

Round-half-away-from-zero

Clamp to int16


8.2 Similarity Metric

Integer cosine-like similarity

Output Q0.15 [-32768, 32767]

Uses deterministic integer sqrt


8.3 Topic Selection

Use topic_hint if provided

Else argmax similarity

Tie-break: lowest topic_id


8.4 Bank Selection

If best POS similarity < 0 → NEG

Else POS

Optional deterministic contradiction hint may override


8.5 Prototype Selection

Only slots with conf > 0

Argmax similarity

Tie-break: lowest proto_idx

Empty bank → index 0


8.6 Action Selection

Condition	Action

Empty slot	REPLACE_ABS
sim ≥ plastic_th	EMA_ADD
sim ≤ novel_th	REPLACE_ABS
otherwise	DRIFT_ADD


8.7 Confidence Mapping

Empty: full confidence

Otherwise: linear map from similarity


8.8 Learning Rate Override

v1 producers must emit lr_q = 0



---

9. NUMCHAIN System Layer (Context)

9.1 Consensus

Leader-based BFT (HotStuff-style)

Static validator set

Ed25519 signatures

Deterministic leader rotation

Quorum = ≥2/3 weight


9.2 Networking

TCP gossip

Reliable ACK + retry for consensus messages

Best-effort gossip for TX propagation

Deterministic message framing


9.3 Persistence

Append-only JSONL log

Replay from genesis

No hidden mutable state


9.4 State Commitment

MA root hash = SHA-256 over all prototypes

Sorted by (topic_id, bank, proto_idx)

Includes vectors, conf, usage



---

10. Security Model (v1)

Assumes trusted validators

Byzantine tolerance limited to consensus layer

No economic security guarantees

Replay safety > liveness > throughput



---

11. Versioning Rules

Any change to:

Binary layout

Arithmetic

Similarity

Tie-breaks

Clamp logic

Producer semantics


→ requires a new protocol version


---

12. Status & Intended Use

This protocol is closed-alpha ready.

Appropriate uses:

Distributed long-term memory experiments

Multi-agent cognitive substrates

Auditable research memory

Swarm or collective knowledge systems


Not appropriate for:

Public permissionless chains

Financial settlement

Adversarial open networks (yet)



---

This document is authoritative.
If an implementation disagrees with this spec, the implementation is wrong.
