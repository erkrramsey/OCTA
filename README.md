# NUMCHAIN v1.4.2 — Memory Arithmetic Chain (Closed Alpha)

NUMCHAIN is an experimental, closed-alpha blockchain kernel built around **Memory Arithmetic (MA)** — a deterministic, fixed-point arithmetic model for evolving sharded tensor state under consensus.

Unlike conventional blockchains that emphasize accounts, balances, or general-purpose virtual machines (EVM/WASM), NUMCHAIN treats **state itself as the primary object**. The chain commits only bounded, verifiable tensor mutations defined by a minimal and deterministic operation set.

This repository contains **v1.4.2**, a single-file Python kernel with lightweight scaffolding for running a multi-node localhost/LAN cluster suitable for closed-alpha testing.

> **Status:** Closed Alpha / Research Kernel  
> **Audience:** Systems engineers, distributed systems researchers, high-TPS experimenters  
> **Explicit non-goal:** Production blockchain, DeFi platform, or public permissionless network

---

## What This Is

NUMCHAIN is:

- A **deterministic, replayable state machine** based on fixed-point tensor arithmetic  
- A **minimal BFT chain kernel** with real cryptographic signatures and multi-node coordination  
- A **research substrate** for exploring distributed, persistent memory evolution  
- A **closed-alpha testnet harness** suitable for 4–10 trusted nodes on localhost or LAN  

Core properties:

- **Memory Arithmetic (MA)** as the sole state transition primitive  
- **Bounded dynamics** (decay + clipping) to prevent unbounded state growth  
- **Sharded state model** (independent tensor partitions)  
- **Ed25519 signatures** (PyNaCl preferred, `cryptography` fallback)  
- **Leader-based BFT-style consensus** with quorum certificates  
- **Transaction gossip with ACK/retry reliability**  
- **Append-only JSONL log with deterministic replay**  
- **Prometheus-compatible `/metrics` endpoint**  
- **One-command local multi-node demo**

---

## What This Is Not

NUMCHAIN is **explicitly not**:

- A general-purpose smart contract platform  
- An EVM / WASM chain  
- A public permissionless network  
- A WAN-hardened P2P system (no libp2p, NAT traversal, or transport security layers)  
- Economically secure or adversarially hardened  
- Optimized for long-term persistence or large validator sets  

If you are looking for:

- DeFi, NFTs, accounts, balances → **this is not that**  
- Production-grade consensus or incentives → **out of scope**  
- Trustless public deployment → **not ready**

---

## Conceptual Model

NUMCHAIN should be understood as a **replicated, consensus-governed memory substrate**, not a financial ledger.

- **State:** A bounded tensor (Memory Arithmetic)  
- **Transactions:** Verified tensor mutations  
- **Blocks:** Synchronization points for agreed mutations  
- **Consensus:** Guarantees identical mutation order across nodes  
- **PoC / PoP (experimental):** Structural alignment of storage/compute contribution with influence over memory evolution  

In this framing, NUMCHAIN is closer to a **distributed cognitive substrate** than a transactional blockchain.

---

## Architecture Overview

### Core Components

- **State:** Sharded MA tensors (`int32`, fixed-point)  
- **Transactions:** Signed MA operations  
- **Consensus:** Minimal leader-based BFT (HotStuff-style phases)  
- **Networking:** TCP gossip (localhost / LAN)  
- **RPC:** FastAPI (transaction submission, queries, metrics)  
- **Persistence:** Append-only JSONL replay log  

### Execution Flow

1. Node boots from shared `genesis.json`  
2. Transactions arrive via RPC or gossip  
3. Leader proposes a block  
4. Validators vote → quorum certificate (QC)  
5. Block commits → MA state mutates  
6. Log entry appended → height advances  
7. Restart → deterministic replay from genesis + log  

---

## Memory Arithmetic (MA)

### State Representation

The full chain state is a fixed-point tensor:

[ shard ][ layer ][ slot ][ dimension ]

All state transitions are **deterministic**, **bounded**, and **replay-safe**.

### Default Parameters (v1.4.2)

| Parameter | Description |
|---------|------------|
| `L = 4` | Layers |
| `S = 64` | Slots per layer |
| `D = 128` | Dimensions |
| `q = 1024` | Fixed-point scale |
| `decay = 0.985` | Exponential decay |
| `clip = 8.0` | Hard magnitude bound |

Persisted state is stored as `int32`.  
`float64` is used **only as a deterministic intermediate** before rounding.

### Supported Operations

Each transaction applies exactly **one** MA operation:

- **`add`** — Add bounded vector to a slot  
- **`decay`** — Apply exponential decay  
- **`mix`** — Linear interpolation between slots  
- **`clip`** — Hard clamp to bounds  

There are:

- No loops  
- No branches  
- No unbounded execution  

This guarantees replay correctness and cross-node determinism.

---

## Transactions

Transactions specify:

- Target shard, layer, and slot  
- Operation type and parameters  
- Ed25519 signature  

Safety properties:

- Size-bounded payloads  
- LRU-based deduplication  
- Per-sender rate caps  
- Gossip propagation with TTL  

---

## Consensus (v1.4.2)

NUMCHAIN implements a **minimal, synchronous-assumption BFT protocol** suitable for closed-alpha coordination.

### Properties

- Static validator set (from genesis)  
- Weighted leader rotation (by bond)  
- 2/3+ quorum certificates  
- Deterministic leader selection  
- RANDAO-style randomness (commit / reveal)  
- Slashing reduces future voting power  

### Important Notes

- Assumes **trusted peers**  
- Gossip reliability uses ACK + bounded retry  
- Byzantine behavior is only partially handled  
- WAN latency and partitions are out of scope  

This is a **correctness-oriented kernel**, not production consensus.

---

## Proof of Capacity (PoC) & Proof of Processing (PoP)

v1.4.2 includes **structural implementations** of PoC and PoP:

- Commit → challenge → verify flow  
- Deterministic verification logic  
- Integrated with consensus weighting  

**Important:**  
These mechanisms demonstrate architectural alignment but **do not yet impose sustained resource pressure or market dynamics**.

---

## Networking & Gossip

- TCP, length-prefixed JSON frames  
- Static peer lists  
- Message types:
  - `PROPOSAL`, `VOTE`, `QC`, `COMMIT`
  - `RANDAO_COMMIT`, `RANDAO_REVEAL`
  - `TXGOSSIP`
- Critical messages use ACK + retry  
- Transaction gossip is best-effort with TTL  

Designed for localhost and LAN clusters only.

---

## RPC Interface

Each node exposes a minimal RPC surface:

- `POST /tx` — Submit transaction  
- `GET /health` — Node status  
- `GET /metrics` — Prometheus metrics  
- `GET /block/{height}` — Block query  
- `GET /receipt/{txid}` — Transaction receipt  

---

## Persistence & Replay

- Append-only JSONL log  
- Records blocks, receipts, and state transitions  
- On restart:
  - Genesis loaded  
  - Log replayed  
  - MA state and bonds reconstructed deterministically  

No database, no snapshots (yet).

---

## Running the Closed-Alpha Cluster

### Requirements

- Python 3.12+  
- Docker (recommended for multi-node demo)  
- Linux or macOS preferred  

### Install Dependencies

```bash
python numchain_v142.py deps

Generate Genesis

python numchain_v142.py genesis --out genesis.json --nodes 5

Run a Single Node

python numchain_v142.py run \
  --genesis genesis.json \
  --node-index 0 \
  --rpc 8000 \
  --gossip 9000 \
  --peers 127.0.0.1:9001,127.0.0.1:9002

Multi-Node Demo (Recommended)

make genesis
make multi

In separate terminals:

make flood-multi
make poll


---

Metrics & Observability

The /metrics endpoint exposes:

Block height

Mempool size

Transaction accept / reject counters

Gossip send / receive / ACK stats

Apply latency p95 / p99

Memory and CPU usage

Slashing events


Fully Prometheus-compatible.


---

Known Limitations

No WAN P2P or NAT traversal

No dynamic validator set

No real economic incentives

No fee market or priority mempool

No state pruning

JSONL replay slows with long chains

Single-file codebase (dense, non-modular)



---

Roadmap (Non-Binding)

Near-term (closed alpha):

Transaction gossip bandwidth tuning

Validator onboarding transactions

Basic state pruning


Longer-term:

WAN-capable P2P layer

Real PoC read pressure

Persistent storage backend

Public testnet



---

Final Note

NUMCHAIN v1.4.2 is not a product — it is a kernel.

Its purpose is to explore a fundamental systems question:

> Can a deterministic, bounded, evolvable memory substrate be replicated and extended by multiple independent machines under consensus?



This repository represents one concrete attempt to answer that question.


---

If needed, this README can be further refined into:

A short (1-page) overview

A validator operator guide

An architecture diagram

A formal whitepaper section

A closed-alpha rollout plan
