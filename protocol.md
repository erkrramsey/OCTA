# NUMCHAIN × MAΔ Protocol (Closed-Alpha Spec)
**Version:** v1 (MAD1 / MAΔ v1)  
**Status:** Closed-alpha / research-grade  
**Scope:** Deterministic distributed memory substrate (NUMCHAIN ordering + MAΔ semantics)  
**Audience:** Implementers, auditors, contributors, early operators  

---

## 0. What This Document Is

This document defines the **NUMCHAIN × MAΔ closed-alpha protocol**, consisting of:

1. **NUMCHAIN** — a deterministic, replayable, fast-finality ordering substrate (small trusted validator set).
2. **MAΔ (Memory Arithmetic Delta)** — a compact, deterministic delta format and semantics for evolving a sharded prototype-memory model (POS/NEG/RES banks), designed to be:
   - **bit-identical under replay**
   - **bounded (no unbounded growth)**
   - **auditable** (hashable, deterministic routing rules)

This spec includes:
- Motivation and architecture
- Threat model and non-goals
- Exact **wire format** for `MAD1` deltas (64-byte header)
- Deterministic fixed-point arithmetic rules (Q formats)
- Deterministic apply rules (integer-only)
- Deterministic producer rules (routing + action selection)
- Conformance requirements and fixtures
- Recommended state commitment (MA root hash)

---

## 1. What This Is

NUMCHAIN × MAΔ is an experimental distributed system for **verifiable, replayable evolution of a compact “cognitive” memory state**.

Instead of account balances or a general-purpose VM, the core state consists of:
- topics
- prototype banks (POS/NEG/RES)
- per-topic, per-bank, per-prototype vectors (Q8.8 fixed-point)
- confidence and usage counters

Updates are encoded as **MADeltas** (MAΔ), ordered and finalized by **NUMCHAIN** consensus. Every correct node should arrive at the **same memory state** from genesis + the ordered delta stream.

---

## 2. What This Is Not

- Not a public permissionless chain (closed-alpha, trusted validators).
- Not a general smart contract VM (no arbitrary code execution).
- Not a WAN-hardened P2P network (no NAT traversal, libp2p, etc. unless added later).
- Not a complete economic system (staking/fees/incentives may be incomplete or toy).
- Not intended to compete with mature L1/L2 systems at this stage.

---

## 3. Architecture Overview

### 3.1 Conceptual Separation
The system is intentionally split into three roles:

- **Producer (Semantics):** Converts raw inputs (embeddings/events) into MAΔ deltas.
- **NUMCHAIN (Ordering/Safety):** Orders, replicates, finalizes deltas.
- **MA Apply (Pure Transition):** Applies deltas deterministically to state.

This separation is critical:
- Producers may differ across implementations, but **compliant producers** must follow canonical rules (Section 9) or match a reference producer (Section 12).
- Apply rules are purely deterministic and **must be bit-identical** across all nodes (Section 8).

### 3.2 High-level Dataflow

```
Raw Input (embedding/event)
        |
        v
  Producer (canonical rules)
        |
        v
   MAD1 delta bytes  --->  NUMCHAIN tx inclusion + finality
        |
        v
  MA Apply (deterministic integer rules)
        |
        v
   Updated MA state + MA root hash
```

---

## 4. Threat Model and Security Posture (Closed-Alpha)

**Assumptions:**
- Validator set is small (e.g., 4–32) and semi-trusted.
- Byzantine behavior is possible but limited; consensus layer may slash/penalize.
- Network is not adversarial at internet scale unless specifically hardened.

**Guaranteed by this spec:**
- Deterministic replay: given identical ordered deltas, all correct nodes converge bit-identically.
- Bounded state updates: per-dimension clip + L2 clamp prevent explosion.
- Hashable state transitions: root commitments can be compared across nodes.

**Not guaranteed yet:**
- WAN security, DoS resistance at scale, permissionless spam resistance.
- Robustness to fully adversarial producers without additional policies.

---

## 5. Versioning & Compatibility

### 5.1 MAΔ Format Versioning
MAΔ deltas use a fixed header with:
- `magic = 0x4D414431` (`MAD1`)
- `version = 1`

Future versions must:
- increment `version`
- preserve old version parsing rules
- never reinterpret existing fields

### 5.2 Forward-compatible fields
- `flags` includes reserved bits.
- `vec_codec` includes reserved encodings (e.g., sparse top-k).
- `topic_tag` reserved for namespacing / derivations.

---

## 6. Core Data Model

### 6.1 Topics
- Topics are indexed by `topic_id` in `[0 .. TOPICS-1]`
- `TOPICS` is fixed by genesis/config for v1.

### 6.2 Banks
- `bank` is an enum:
  - `1 = POS`
  - `2 = NEG`
  - `3 = RES` (reserved; optional in v1 producer)

Each topic contains up to `K_POS` POS prototypes and `K_NEG` NEG prototypes (and optionally RES).

### 6.3 Prototype Record
Each prototype slot stores:
- `vec`: `DIM` components, each `int32` interpreted as **Q8.8**
- `conf`: `u16` interpreted as **Q4.12**
- `usage`: `u32` saturating counter

Genesis/config must enforce:
- `K_POS <= 256` and `K_NEG <= 256` (proto_idx is u8)
- `DIM` is fixed and must match deltas

---

## 7. Fixed-Point Arithmetic & Representation

### 7.1 Vector Format: Q8.8
All vector components are interpreted as **Q8.8** fixed-point:

- Payload vectors: `int16` components, interpreted as Q8.8.
- Internal stored vectors: `int32` components, interpreted as Q8.8.
- Promotion rule: **sign-extend only** (no shifting).

**Example:** payload `i16 = 256` represents `+1.0`.

### 7.2 Confidence: Q4.12
`conf_q` uses `u16` interpreted as Q4.12 (0..~15.999).  
In producer reference, we clamp to `[0..4096]` as a v1 policy; apply may saturate to `[0..65535]` if desired.

### 7.3 Learning Rate: Q0.16
`lr_q` uses `u16` interpreted as Q0.16.  
`0` means: “use genesis default”.

### 7.4 Deterministic Float → Fixed (Producer Side)
When producers convert float embeddings to Q8.8:

- Use `round-half-away-from-zero`:
  - if `scaled >= 0`: `int(scaled + 0.5)`
  - else: `int(scaled - 0.5)`
- Clamp to `[-32768, 32767]`

This rule is mandatory for canonical producers.

---

## 8. MAΔ Apply Semantics (Deterministic State Transition)

### 8.1 Design Choice (v1)
**No normalization** is performed in v1.  
Instead, vectors are bounded by:
1) mandatory per-dimension clip  
2) deterministic L2 clamp  

This avoids floating point and minimizes cross-platform divergence risk.

### 8.2 Genesis/Config Bounding Constants
Genesis/config must include:

- `per_dim_clip_q` (Q8.8, int32): mandatory (recommended `5120` ≈ 20.0)
- `max_norm_q` (Q8.8, int32): mandatory (recommended `2560` ≈ 10.0)
- `default_lr_q` (Q0.16, u16)
- `gentle_lr_q` (Q0.16, u16)

### 8.3 MAD1 Actions
`action` enum:
- `0 = EMA_ADD`
- `1 = REPLACE_ABS`
- `2 = DRIFT_ADD`
- `3 = CONF_ONLY`

Interpretation:

- **CONF_ONLY:** no payload allowed; only confidence updates (usage increments).
- **REPLACE_ABS:** payload is absolute prototype vector.
- **EMA_ADD / DRIFT_ADD:** payload is a delta vector, scaled by `lr_q` and added to existing.

### 8.4 Apply Algorithm (Normative, Integer-only)

#### 8.4.1 Parsing constraints
- Header must be exactly 64 bytes.
- `magic == 0x4D414431`
- `version == 1`
- `dim == GENESIS.dim`
- `vec_codec == 0` (dense i16 Q8.8 only in v1)
- `proto_idx < K_bank`
- If `action == CONF_ONLY`, `payload_len == 0`.
- Otherwise, `payload_len == dim * 2`.

#### 8.4.2 Learning rate selection
- If `action == EMA_ADD`:
  - if `lr_q != 0`, use it
  - else use `default_lr_q`
- If `action == DRIFT_ADD`:
  - always use `gentle_lr_q` (override not permitted)
- If `action == REPLACE_ABS`:
  - `lr_q` ignored

#### 8.4.3 Update logic
Given existing prototype `(vec, conf, usage)` possibly empty.

- If slot is empty:
  - only `REPLACE_ABS` is valid
  - write payload as vec
- If slot exists:
  - `REPLACE_ABS`: `vec = payload`
  - `EMA_ADD` or `DRIFT_ADD`:
    - for each component:
      - `ds = (payload_i16 * lr_q) >> 16`   (still Q8.8)
      - `vec_i32 = vec_i32 + ds`

#### 8.4.4 Mandatory per-dimension clip (before L2 clamp)
For all vector-modifying actions:
- `vec[i] = clamp(vec[i], -per_dim_clip_q, +per_dim_clip_q)`

#### 8.4.5 Deterministic L2 clamp (after per-dim clip)
Enforce:
- `||vec||₂ <= max_norm_q`

Computation is integer-only using `isqrt_u64` (below).
Zero vectors are preserved.

Pseudo-logic:
- `sumsq = Σ (vec[i]^2)` (use int64 accumulators)
- `curr = isqrt_u64(sumsq)` yields Q8.8 magnitude because sqrt(Q16.16) → Q8.8
- If `curr <= max_norm_q`: keep vec
- Else:
  - `scale_q = (max_norm_q << 16) / curr` (Q0.16)
  - `vec[i] = (vec[i] * scale_q) >> 16`

#### 8.4.6 Final safety clamp
After scaling:
- `vec[i] = clamp(vec[i], -32768*256, +32767*256)`  
This ensures values remain i16-compatible range if needed downstream.

#### 8.4.7 Confidence + usage updates
- `conf = saturating_add(conf, delta_conf_q)`
- `usage = saturating_add(usage, 1)`
- If `action == REPLACE_ABS` and slot existed:
  - optional policy: `usage = max(1, usage // 2)` (recommended)

### 8.5 Deterministic isqrt_u64 (Normative)
All implementations MUST match this exact floor-sqrt behavior:

```rust
/// Floor square root for u64 values (0 ≤ n < 2^64).
/// Deterministic across platforms.
pub fn isqrt_u64(mut n: u64) -> u64 {
    if n == 0 { return 0; }

    let mut x: u64 = 0;
    let mut bit: u64 = 1 << 62;

    while bit > n { bit >>= 2; }

    while bit != 0 {
        if n >= x.wrapping_add(bit) {
            n = n.wrapping_sub(x.wrapping_add(bit));
            x = (x >> 1).wrapping_add(bit);
        } else {
            x >>= 1;
        }
        bit >>= 2;
    }
    x
}
```

---

## 9. Producer Semantics (Canonical Producer Rules, v1)

The producer is responsible for generating deltas from inputs in a deterministic way.  
Without canonical producer rules, “semantic divergence” occurs even if apply is deterministic.

### 9.1 Similarity Metric (Normative)
Producers use an integer cosine-like similarity returning Q0.15 (`[-32768..32767]`).

### 9.2 Topic Selection
If `topic_hint` is provided (non-protocol, app-level hint), it overrides.

Otherwise:
- Compute similarity between input embedding and each topic mean (or canonical routing vector).
- Select:
  - `topic_id = argmax(sim)`
  - Tie-break: **lowest topic_id** wins.

### 9.3 Contradiction Routing (POS vs NEG)
Let `best_pos_sim` be the maximum similarity vs any POS prototype in the selected topic.

- If contradiction_hint is provided (app-level), it overrides.
- Else:
  - `is_contradiction = (best_pos_sim < 0)`
  - `bank = NEG if is_contradiction else POS`

### 9.4 Prototype Index Selection
Within the chosen `(topic_id, bank)`:
- Consider prototypes with `conf > 0`.
- Choose:
  - `proto_idx = argmax(similarity(input, proto_vec))`
  - Tie-break: **lowest proto_idx** wins.
- If no prototypes exist/populated:
  - treat bank as empty and select `proto_idx = 0`.

### 9.5 Action Selection (Plasticity / Novelty Gating)
Let `best_sim` be the similarity score for the selected `proto_idx`.

- If bank empty:
  - `action = REPLACE_ABS`
- Else:
  - If `best_sim >= PLASTIC_TH_SIM_Q15`:
    - `action = EMA_ADD`
  - Else if `best_sim <= NOVEL_TH_SIM_Q15`:
    - `action = REPLACE_ABS`
  - Else:
    - `action = DRIFT_ADD`

### 9.6 delta_conf_q Mapping
v1 recommended mapping:
- If empty bank:
  - `delta_conf_q = 4096`
- Else:
  - `delta_conf_q = clamp( (max(best_sim,0) * 4096) / 32767, 0, 4096 )`

### 9.7 lr_q Override
v1 canonical producer outputs:
- `lr_q = 0` always (use defaults).

---

## 10. MAΔ Binary Format (MAD1)

All MAD1 deltas are encoded as:

```
[ 64-byte header ][ payload bytes ][ optional 32-byte provenance hash ]
```

### 10.1 Header Layout (Little-endian, Fixed 64 bytes)

| Offset | Type | Field | Size | Notes |
|-------:|------|-------|------:|------|
| 0 | u32 | magic | 4 | 0x4D414431 ("MAD1") |
| 4 | u16 | version | 2 | = 1 |
| 6 | u16 | flags | 2 | bitfield |
| 8 | u64 | chain_id_hash | 8 | cross-chain protection |
| 16 | u64 | height_hint | 8 | 0 if unused |
| 24 | u64 | timestamp_ns | 8 | leader-assigned monotonic |
| 32 | u32 | topic_id | 4 | topic index |
| 36 | u64 | topic_tag | 8 | reserved |
| 44 | u8 | bank | 1 | 1 POS, 2 NEG, 3 RES |
| 45 | u8 | action | 1 | 0 EMA, 1 REPLACE, 2 DRIFT, 3 CONF |
| 46 | u8 | vec_codec | 1 | 0 dense i16 Q8.8 |
| 47 | u8 | proto_idx | 1 | 0..255 |
| 48 | u16 | dim | 2 | must match genesis |
| 50 | u16 | reserved | 2 | must be 0 |
| 52 | u16 | delta_conf_q | 2 | Q4.12 |
| 54 | u16 | lr_q | 2 | Q0.16 (0 = default) |
| 56 | u32 | payload_len | 4 | exact bytes following |
| 60 | pad | padding | 4 | must exist (header=64) |

### 10.2 Flags
- bit 0: `has_provenance` (32-byte SHA-256 tail)
- bits 1..15: reserved, must be 0 for v1

### 10.3 Payload (vec_codec = 0)
- If `action == CONF_ONLY`:
  - payload_len must be 0
- Else:
  - payload is `dim` little-endian `int16` values (Q8.8)
  - payload_len must equal `dim * 2`

### 10.4 Size
For `dim=256`:
- no provenance: `64 + 512 = 576 bytes`
- with provenance: `64 + 512 + 32 = 608 bytes`

---

## 11. State Commitment (MA Root Hash)

Implementations SHOULD provide a deterministic commitment hash of MA state after each block.

---

## 12. Reference Producer and Conformance

### 12.1 Reference Producer
A single-file, stdlib-only Python producer is the **gold standard** for v1 canonical producer behavior.

**Normative requirement:** Any compliant producer MUST match the reference producer output SHA-256 on fixture inputs.

### 12.2 Fixtures (Mandatory)
The reference producer should include fixture cases for:
- empty bank replace
- EMA path
- contradiction → NEG routing
- drift path
- novelty replace
- (optional) CONF_ONLY

### 12.3 Conformance Tests
A producer is compliant if:
- It matches the fixture SHA-256 values exactly.
- It uses deterministic float→fixed rounding as specified.
- It uses deterministic tie-break rules and similarity metric.

An apply implementation is compliant if:
- Given identical ordered deltas, it produces identical MA root hash across platforms.
- It implements per-dim clip and L2 clamp with normative isqrt_u64.

---

## 13. Integration Notes (NUMCHAIN)

This protocol assumes NUMCHAIN provides:
- deterministic transaction ordering (fast finality BFT in closed-alpha)
- optional gossip transport for tx propagation
- log replay from genesis (append-only journal acceptable)

---

## 14. Operational Guidance (Closed Alpha)

### 14.1 Validation Checklist
Before shipping closed-alpha:
1. Reference producer fixture locking (`--gen-fixtures`, then `--test`)
2. Round-trip byte stability (same input → same SHA-256)
3. Cross-platform producer check (different Python/OS/arch)
4. Kernel apply convergence test (3–5 nodes → identical MA root hash)

---

## 15. Known Limitations (v1)

- No orthospace head projections in canonical producer (planned extension).
- No sparse encoding in v1 (vec_codec reserved).
- No dynamic topic creation rules specified (topics fixed).
- Not WAN hardened.
- Producer state (topic means / prototypes) requires chain queries or local mirrored state.

---

## 16. Future Extensions (Non-breaking Paths)

Potential v1.1/v2 additions:
- orthospace multi-head similarity (deterministically derived matrices)
- sparse top-k payload codec
- RES bank semantics for residuals
- topic_tag namespacing
- dynamic topic creation/merge rules
- fee/priority scheduling for deltas

---

## Appendix: Suggested Default Parameters (v1)

Recommended:
- `DIM = 256`
- `TOPICS = 64`
- `K_POS = 6`
- `K_NEG = 3`
- `per_dim_clip_q = 5120`  (≈ 20.0)
- `max_norm_q = 2560`      (≈ 10.0)
- `default_lr_q = 6554`    (≈ 0.10)
- `gentle_lr_q = 655`      (≈ 0.01)
- `plastic_th_sim_q15 = 14746` (≈ 0.45)
- `novel_th_sim_q15 = 11469`   (≈ 0.35)
