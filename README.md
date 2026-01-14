# OCTA
OCTA AGI Labs

---

NUMCHAIN v1.4.2 — Memory Arithmetic Chain (Closed Alpha)

NUMCHAIN is an experimental, closed-alpha blockchain-like system centered around Memory Arithmetic (MA) — a deterministic, fixed-point arithmetic model for evolving sharded tensor state under consensus.

Unlike traditional blockchains that focus on accounts, tokens, or general virtual machines (EVM/WASM), NUMCHAIN treats state itself as the primary object: a bounded, verifiable, replayable memory substrate mutated only through a minimal set of deterministic operations.

This repository contains v1.4.2, a single-file Python kernel plus lightweight scaffolding for running a multi-node localhost/LAN cluster.

> Status: Closed-alpha prototype
Audience: Systems engineers, distributed systems researchers, high-TPS experimenters
Non-Goal: This is not a production blockchain, DeFi platform, or public testnet.




---

What This Is

NUMCHAIN is:

A deterministic, replayable state machine built on fixed-point tensor arithmetic

A minimal BFT chain kernel with real cryptographic signatures and multi-node coordination

A research substrate for exploring distributed, persistent memory evolution

A closed-alpha testnet harness suitable for 4–10 trusted nodes on localhost or LAN


Key properties:

Memory Arithmetic (MA) as the only state transition primitive

Bounded dynamics (decay + clipping) → no unbounded state explosion

Sharded state (independent tensor partitions)

Ed25519 signatures (PyNaCl preferred, cryptography fallback)

Leader-based BFT-style consensus with quorum certificates

Transaction gossip with ACK/retry reliability

Append-only JSONL log + deterministic replay

Prometheus-ready /metrics endpoint

One-command local multi-node demo



---

What This Is Not

NUMCHAIN is explicitly not:

A general-purpose smart contract platform

An EVM/WASM chain

A public permissionless network

A WAN-hardened P2P system (no libp2p, NAT traversal, encryption layers)

Economically secure or attack-resistant

Optimized for long-term persistence or large validator sets


If you are looking for:

DeFi, NFTs, accounts, balances → wrong project

Production consensus or incentives → too early

Trustless adversarial security → not yet



---

Mental Model: How to Think About NUMCHAIN

NUMCHAIN is best understood as a replicated, consensus-governed memory substrate.

State: A bounded tensor (Memory Arithmetic)

Transactions: Verified mutations of that tensor

Blocks: Synchronization points for agreed mutations

Consensus: Ensures all nodes apply the same mutations in the same order

PoC / PoP (experimental): Structural alignment of storage/compute contribution with influence over memory evolution


In this framing, NUMCHAIN is closer to a distributed cognitive substrate than a financial ledger.


---

Architecture Overview

High-Level Components

State: Sharded MA tensors (int32, fixed-point)

Transactions: Signed MA operations

Consensus: Minimal leader-based BFT (HotStuff-style phases)

Networking: TCP gossip (localhost/LAN)

RPC: FastAPI (submit tx, query blocks, metrics)

Persistence: Append-only JSONL replay log


Data Flow

1. Node boots from shared genesis.json


2. Transactions arrive via RPC or gossip


3. Leader proposes block


4. Validators vote → quorum certificate (QC)


5. Block commits → MA state mutates


6. Log entry written → height advances


7. Restart → full deterministic replay




---

Memory Arithmetic (MA)

Core Idea

The entire chain state is a fixed-point tensor:

[ shard ][ layer ][ slot ][ dimension ]

All state transitions are deterministic, bounded, and verifiable.

Parameters (Default v1.4.2)

Parameter	Meaning

L = 4	Layers
S = 64	Slots per layer
D = 128	Dimensions
q = 1024	Fixed-point scale
decay = 0.985	Exponential decay
clip = 8.0	Hard magnitude bound


All persisted state is int32.
Float64 is used only as a deterministic intermediate before rounding.

Supported Operations

Each transaction applies exactly one MA operation:

add: Add bounded vector to a slot

decay: Multiply slot by decay factor

mix: Linear interpolation between slots

clip: Hard clamp to bounds


No loops.
No branches.
No unbounded execution.

This guarantees replay safety and cross-node determinism.


---

Transactions

A transaction specifies:

Target shard / layer / slot

Operation type

Parameters (vector, alpha, decay, etc.)

Ed25519 signature


Transactions are:

Size-bounded

Deduplicated (LRU)

Rate-limited per sender

Gossip-propagated with TTL



---

Consensus (v1.4.2)

NUMCHAIN uses a minimal, synchronous-assumption BFT-style protocol suitable for closed-alpha testing.

Properties

Static validator set (from genesis)

Weighted leader rotation (by bond)

2/3+ quorum certificates

Deterministic leader selection

RANDAO-style randomness (commit/reveal)

Slashing reduces future voting power


Important Notes

All networking assumes trusted peers

Gossip reliability uses ACK + bounded retry

Byzantine behavior is not fully tolerated

WAN latency and partitions are out of scope


This is not production consensus — it is a correctness-oriented kernel.


---

Proof of Capacity (PoC) & Proof of Processing (PoP)

v1.4.2 includes structural implementations of PoC and PoP:

Commit → challenge → verify flow

Deterministic verification

Integrated with consensus weighting


However:

> PoC/PoP are architecturally correct but do not yet impose sustained resource pressure or economic market dynamics.



They exist to show how storage/compute can align with MA evolution — not to provide real security yet.


---

Networking & Gossip

TCP, length-prefixed JSON frames

Static peer list

Message types:

PROPOSAL, VOTE, QC, COMMIT

RANDAO commit/reveal

TXGOSSIP


Critical messages use ACK + retry

TX gossip is best-effort with TTL


This is sufficient for:

Localhost clusters

LAN demos

Closed-alpha coordination


Not suitable for WAN.


---

RPC API

Each node exposes a minimal RPC surface:

POST /tx — submit transaction

GET /health — node status

GET /metrics — Prometheus metrics

GET /block/{height} — block query

GET /receipt/{txid} — tx receipt



---

Persistence & Replay

Append-only JSONL log

Records blocks, tx receipts, state transitions

On restart:

Genesis loaded

Log replayed

MA state and bonds reconstructed deterministically



No database.
No snapshots (yet).


---

Running the Closed-Alpha Cluster

Requirements

Python 3.12+

Docker (for multi-node demo)

Linux / macOS recommended


Install Dependencies

python numchain_v142.py deps

Generate Genesis

python numchain_v142.py genesis --out genesis.json --nodes 5

Run Single Node

python numchain_v142.py run \
  --genesis genesis.json \
  --node-index 0 \
  --rpc 8000 \
  --gossip 9000 \
  --peers 127.0.0.1:9001,127.0.0.1:9002

Multi-Node (Recommended)

Use the provided scaffold:

make genesis
make multi

Then in another terminal:

make flood-multi
make poll


---

Metrics & Observability

The /metrics endpoint exposes:

Block height

Mempool size

TX accept/reject counters

Gossip send/recv/ack stats

Apply latency p95/p99

Memory usage

Slashing events


Grafana-ready.


---

Known Limitations

No WAN P2P

No dynamic validator set

No real economic incentives

No mempool fee market

No state pruning

JSONL replay slows with long chains

Single-file codebase (dense, not modular)



---

Roadmap (Non-Binding)

Near-term (closed alpha):

TX gossip bandwidth tuning

Validator onboarding tx

Basic state pruning


Longer-term:

WAN P2P

Real PoC read pressure

Persistent storage backend

Public testnet



---

Final Note

NUMCHAIN v1.4.2 is not a product — it is a kernel.

Its purpose is to answer a hard systems question:

> Can we build a deterministic, bounded, evolvable memory substrate that multiple independent machines can agree on and extend over time?



This repository is one concrete attempt at that answer.


---
