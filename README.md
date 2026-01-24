
---

# Orthospace Afocal Tumbler Imaging Model (DoD Reference Architecture)

Orthospace is a **multi–field-of-view (Multi-FOV), dual-band imaging architecture** for modular optical systems that use a **rotating electro-mechanical tumbler** to switch between discrete optical configurations.

It provides:
- Multiple selectable FOVs on an existing dual-band host imager
- Two **orthogonal** afocal optical paths for complementary projections
- An integrated **uniform-temperature calibration source**
- A configuration-aware processing model for calibration, normalization, and fusion

This repository is written as a **DoD program reference architecture**: integration-oriented, implementation-agnostic, and suitable for government/contractor collaboration.

---

## Design Origin and Patent Lineage

Orthospace is derived from the design principles disclosed in a **United States Government patent** describing a *multiple field-of-view, dual-band optical device with an integrated calibration source*.

The patented concept discloses a **modular optical device** intended to be inserted in front of a camera system that already contains:
- an imager optical assembly, and
- a detector focal plane array (FPA).

The module includes:
- a rotating electro-mechanical **tumbler**
- two **orthogonal Galilean afocal** optical assemblies
- at least one **uniform temperature calibration** source

The purpose of the module is to provide an existing dual-band camera with **up to four additional selectable FOVs** and an integrated calibration reference, all selectable by the host imager.

Orthospace generalizes this patented design into a **configuration-aware system model** that supports modern calibration, drift monitoring, and multi-FOV fusion.

---

## Intended Use and Government Program Availability

This document and the Orthospace architecture it describes are intended for **free and open use within United States Department of Defense (DoD) programs**, government laboratories, FFRDCs, and associated contractors.

It is provided to support:
- system design and analysis
- RDT&E
- integration with defense sensor platforms
- open technical collaboration across government and industry teams

No proprietary implementation details are included. This is **unrestricted technical reference material**.

---

## System Summary (Plain Language)

Orthospace is a front-mounted module that sits in front of a dual-band camera.  
It enables rapid mode switching between multiple optical states, while keeping the host camera unchanged.

Key properties:
- **Discrete, repeatable configurations** (mechanically indexed)
- **Orthogonal optical axes** (two afocal paths at right angles)
- **Integrated calibration** (uniform-temperature source visible through modes)
- **Mode labeling** (every frame is tagged with its configuration)

---

## DoD Reference Architecture Diagrams

### 1) Physical Integration View

[ SCENE / TARGET ] │ ▼ ┌──────────────────────────────┐ │  ORTHOSPACE AFACAL MODULE     │ │  - Rotating tumbler (indexed) │ │  - Afocal path A (axis 1)     │ │  - Afocal path B (axis 2)     │ │  - Uniform-temp cal source    │ └──────────────┬───────────────┘ │ ▼ ┌──────────────────────────────┐ │  HOST DUAL-BAND CAMERA        │ │  - Imager optics             │ │  - Dual-band FPA             │ └──────────────┬───────────────┘ │ ▼ ┌──────────────────────────────┐ │  PROCESSING PIPELINE          │ │  - Mode ID / metadata         │ │  - Calibration / normalization│ │  - Multi-FOV fusion           │ │  - Drift monitoring           │ └──────────────────────────────┘

---

### 2) Configuration/Mode Model (Operational)

Tumbler Index i  ─┐ Optical Path p   ─┼──►  Configuration Tag  (i, p, b) Spectral Band b  ─┘

Every frame is processed with its configuration tag. This enables deterministic calibration and fusion across modes.

---

### 3) Calibration Anchor Concept

┌─────────────────────────┐
           │ Uniform-temp source (T) │
           └───────────┬─────────────┘
                       │ Visible in each mode
                       ▼

Mode i1 ──► estimate gain/offset ──► residual stats Mode i2 ──► estimate gain/offset ──► residual stats Mode i3 ──► estimate gain/offset ──► residual stats

Result: continuous drift monitoring across all modes and both bands.

---

### 4) Fusion Pipeline (Implementation-Agnostic)

Multi-FOV dual-band frames (tagged by mode) │ ▼ Geometry alignment / registration │ ▼ Radiometric normalization (cal source anchor) │ ▼ Common scene representation (latent) │ ▼ Detection / track / classification / exploitation

---

## Core Technology Basis (What Makes It Work)

1. **Electro-Mechanical Configuration Indexing**
   - Each tumbler state is repeatable and addressable
   - Software always knows “which optics were used”

2. **Orthogonal Afocal Assemblies**
   - Two right-angle afocal axes provide complementary projections
   - Reduces ambiguity and improves fusion conditioning

3. **Dual-Band Operation**
   - Both bands share the same mode structure
   - Enables consistent multi-band exploitation

4. **Integrated Calibration Source**
   - Provides a stable “anchor” in every mode
   - Supports gain/offset estimation and drift detection over time

---

## Military and Defense Applications (Representative)

### ISR (Intelligence, Surveillance, Reconnaissance)
- Wide-area search + narrow-field inspection without swapping sensors
- Dual-band exploitation (e.g., visible/IR) under a unified mode model
- More robust detection due to complementary orthogonal projections

### Target Acquisition / Support to Fire Control
- Deterministic optical modes for consistent observation geometry
- Stable thermal signature interpretation from calibrated radiometry
- Reduced mode-to-mode registration ambiguity

### Persistent Surveillance / Long-Duration Operations
- Integrated calibration supports long deployment without frequent external calibration
- Drift monitoring enables maintenance planning and QA

### Platform Integration (Air/Ground/Maritime)
- Front-mounted modular approach reduces integration risk
- Host camera remains unchanged
- Mode selection is software-controlled

---

## TRL Framing (DoD-Aligned)

Orthospace supports accelerated transition through the DoD TRL scale:

- **TRL 2–3 (Concept / Analytical Validation):**
  - Mode indexing model, orthogonal projection rationale, calibration anchor concept
- **TRL 4 (Component Validation in Lab):**
  - Afocal paths and calibration source validated on bench with a dual-band host camera
- **TRL 5 (Relevant Environment Validation):**
  - Prototype tumbler module integrated with representative host camera; environmental testing
- **TRL 6 (System/Sub-System Demonstration):**
  - End-to-end prototype demonstrated in mission-representative conditions with fusion and drift monitoring

---

## Scope and Boundaries

This repository contains **architectural and technical descriptions only**.
It does not include:
- hardware drawings or manufacturing data
- operational procedures
- controlled technical data
- classified content

---

## License and Use

This document is provided as **open technical reference material** for United States Government and DoD program use.

- No proprietary claims are made.
- No restriction is placed on government, academic, or contractor use.
- Implementations may be subject to separate program, contract, export control, or security requirements.

This document does **not** convey classified or export-controlled information.

Use is intended to be **freely permitted within DoD programs and associated partners**.


---

