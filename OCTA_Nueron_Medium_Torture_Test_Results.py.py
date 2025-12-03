
#!/usr/bin/env python3
"""
OCTA NEURON v9 — Torture Test Harness

This script stress-tests the OCTA Neuron v9 with Perfect-DNA Attractor in two regimes:

1) APOPTOSIS TORTURE
   - Constant negative survival reward pulses
   - Low external drive (neuron is "punished" and bored)
   - Compare:
       A) DNA ON   (default dna scales)
       B) DNA OFF  (dna_homeo_scale = dna_ATP_scale = dna_meta_scale = 0)

   We expect:
       - apoptosis_risk to climb in both
       - possibly earlier / more decisive apoptosis in DNA-ON if life is truly bad
         (it stabilizes normal regimes but should allow death under persistent pain)

2) ENERGY STRESS (ATP TORTURE)
   - High external drive (lots of spikes)
   - Neutral rewards (no big positive/negative)
   - Compare:
       A) DNA ON   (default dna scales)
       B) DNA OFF  (dna_* scales = 0)

   We expect:
       - DNA ON to keep ATP higher / more stable than DNA OFF
       - Both should stay alive for this short run, but DNA OFF may run lower ATP.

To run:

    python3 octa_neuron_v9_torture.py

You will get printed summaries for all four runs.
"""

import math
from typing import Callable, Dict, Any, Tuple

import numpy as np

from octa_neuron_v9_perfect_dna import OCTANeuron, OCTANeuronConfig


# ---------------------------------------------------------------------------
# Generic run helper
# ---------------------------------------------------------------------------

def run_sim(
    label: str,
    cfg: OCTANeuronConfig,
    steps: int,
    I_ext_fn: Callable[[int], float],
    reward_fn: Callable[[int, OCTANeuron], None],
) -> Dict[str, Any]:
    """
    Run one neuron simulation and log key quantities.

    Returns dict:
        {
            "label": str,
            "time_ms": np.ndarray [T],
            "V_soma": np.ndarray [T],
            "ATP": np.ndarray [T],
            "apoptosis_risk": np.ndarray [T],
            "dna_stability": np.ndarray [T],
            "alive": np.ndarray [T],
            "spike_count": int,
            "final_summary": dict,
        }
    """
    neuron = OCTANeuron(cfg)
    T = steps

    time_ms = np.zeros(T, dtype=np.float32)
    V_soma = np.zeros(T, dtype=np.float32)
    ATP = np.zeros(T, dtype=np.float32)
    apoptosis_risk = np.zeros(T, dtype=np.float32)
    dna_stability = np.zeros(T, dtype=np.float32)
    alive = np.zeros(T, dtype=bool)

    spikes = 0

    for step in range(T):
        if not neuron.alive:
            # Freeze state after death (flatline)
            time_ms[step:] = neuron.time_ms
            V_soma[step:] = neuron.V_soma
            ATP[step:] = neuron.ATP
            apoptosis_risk[step:] = neuron.apoptosis_risk
            stab = neuron.dna_last_out.get("stability", 0.0)
            dna_stability[step:] = stab
            alive[step:] = False
            break

        # Reward schedule
        reward_fn(step, neuron)

        # External current schedule
        I_ext = I_ext_fn(step)

        # Step
        if neuron.step(I_ext=I_ext):
            spikes += 1

        # Log
        time_ms[step] = neuron.time_ms
        V_soma[step] = neuron.V_soma
        ATP[step] = neuron.ATP
        apoptosis_risk[step] = neuron.apoptosis_risk
        dna_stability[step] = neuron.dna_last_out.get("stability", 0.0)
        alive[step] = neuron.alive

    summary = neuron.summary()

    return {
        "label": label,
        "time_ms": time_ms,
        "V_soma": V_soma,
        "ATP": ATP,
        "apoptosis_risk": apoptosis_risk,
        "dna_stability": dna_stability,
        "alive": alive,
        "spike_count": spikes,
        "final_summary": summary,
    }


def print_run_stats(res: Dict[str, Any]) -> None:
    """Pretty-print key stats for a run."""
    label = res["label"]
    t = res["time_ms"]
    V = res["V_soma"]
    ATP = res["ATP"]
    risk = res["apoptosis_risk"]
    stab = res["dna_stability"]
    alive = res["alive"]
    spikes = res["spike_count"]
    fs = res["final_summary"]

    duration_ms = float(t[len(t) - 1])
    alive_last = bool(alive[len(alive) - 1])

    ATP_min = float(np.min(ATP))
    ATP_mean = float(np.mean(ATP))
    risk_max = float(np.max(risk))
    risk_final = float(risk[len(risk) - 1])
    stab_mean = float(np.mean(stab))
    stab_final = float(stab[len(stab) - 1])

    print("\n" + "=" * 70)
    print(f"RUN: {label}")
    print("=" * 70)
    print(f"Duration:        {duration_ms:.2f} ms  (steps = {len(t)})")
    print(f"Alive final:     {alive_last}")
    print(f"Spike count:     {spikes}")
    print(f"V_soma range:    [{float(np.min(V)):.2f}, {float(np.max(V)):.2f}] mV")
    print(f"ATP min/mean:    {ATP_min:.3f} / {ATP_mean:.3f}")
    print(f"Apoptosis risk:  max = {risk_max:.6f}, final = {risk_final:.6f}")
    print(f"DNA stability:   mean = {stab_mean:.4f}, final = {stab_final:.4f}")
    print("Final summary snapshot (selected):")
    for k in [
        "alive",
        "apoptosis_risk",
        "V_soma",
        "ATP",
        "homeo_rate_est",
        "recent_rate_hz",
        "spike_count",
        "dna_stability",
        "dna_error_norm",
    ]:
        if k in fs:
            v = fs[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Experiment 1: Apoptosis torture
# ---------------------------------------------------------------------------

def make_apoptosis_configs(seed_base: int = 100) -> Tuple[OCTANeuronConfig, OCTANeuronConfig]:
    """
    Return (cfg_dna_on, cfg_dna_off) for apoptosis torture.

    Both identical except DNA scaling is zeroed in the OFF case.
    """
    base = OCTANeuronConfig(
        seed=seed_base,
        use_adex=False,
        num_dendrites=4,
        meta_dim=16,
    )

    # DNA ON: use defaults
    cfg_on = base

    # DNA OFF: neutralize all DNA couplings
    cfg_off = OCTANeuronConfig(
        **{**base.__dict__},  # copy
    )
    cfg_off.dna_homeo_scale = 0.0
    cfg_off.dna_ATP_scale = 0.0
    cfg_off.dna_meta_scale = 0.0

    return cfg_on, cfg_off


def apoptosis_torture_reward(step: int, neuron: OCTANeuron) -> None:
    """
    Chronic negative survival reward.

    Every 50 steps (~2.5 ms at dt=0.05ms), we punish it.
    Very rough "this universe sucks" signal.
    """
    if step % 50 == 0:
        neuron.register_reward(survival=-1.0)


def apoptosis_torture_I_ext(step: int) -> float:
    """
    Low external drive – the neuron is under-stimulated and punished.
    """
    return 3.0  # small positive input


# ---------------------------------------------------------------------------
# Experiment 2: Energy (ATP) torture
# ---------------------------------------------------------------------------

def make_energy_configs(seed_base: int = 200) -> Tuple[OCTANeuronConfig, OCTANeuronConfig]:
    """
    Return (cfg_dna_on, cfg_dna_off) for ATP torture.

    Both identical except DNA scaling is zeroed in the OFF case.
    """
    base = OCTANeuronConfig(
        seed=seed_base,
        use_adex=False,
        num_dendrites=4,
        meta_dim=16,
        ATP_spike_cost=0.04,      # more expensive spikes
        ATP_recovery_rate=0.0025, # slightly faster recovery
    )

    cfg_on = base

    cfg_off = OCTANeuronConfig(
        **{**base.__dict__},
    )
    cfg_off.dna_homeo_scale = 0.0
    cfg_off.dna_ATP_scale = 0.0
    cfg_off.dna_meta_scale = 0.0

    return cfg_on, cfg_off


def energy_torture_reward(step: int, neuron: OCTANeuron) -> None:
    """
    Neutral / small positive reward, no big punishment or blessing.
    """
    if step % 1000 == 0:
        neuron.register_reward(survival=0.2)  # barely noticeable


def energy_torture_I_ext(step: int) -> float:
    """
    High external drive, to force lots of spiking and ATP burn.
    """
    return 20.0  # strong input


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Simulation duration: 20_000 steps * 0.05 ms = 1000 ms (1 second)
    steps_apoptosis = 20_000
    steps_energy = 20_000

    print("==== OCTA NEURON v9 TORTURE TESTS ====")

    # ----------------- APOPTOSIS TORTURE -----------------
    print("\n--- Experiment 1: Apoptosis torture (chronic negative reward) ---")
    cfg_on, cfg_off = make_apoptosis_configs()

    res_apoptosis_on = run_sim(
        label="Apoptosis torture – DNA ON",
        cfg=cfg_on,
        steps=steps_apoptosis,
        I_ext_fn=apoptosis_torture_I_ext,
        reward_fn=apoptosis_torture_reward,
    )

    res_apoptosis_off = run_sim(
        label="Apoptosis torture – DNA OFF",
        cfg=cfg_off,
        steps=steps_apoptosis,
        I_ext_fn=apoptosis_torture_I_ext,
        reward_fn=apoptosis_torture_reward,
    )

    print_run_stats(res_apoptosis_on)
    print_run_stats(res_apoptosis_off)

    # ----------------- ENERGY (ATP) TORTURE -----------------
    print("\n--- Experiment 2: Energy (ATP) torture (high activity) ---")
    cfg_on_e, cfg_off_e = make_energy_configs()

    res_energy_on = run_sim(
        label="Energy torture – DNA ON",
        cfg=cfg_on_e,
        steps=steps_energy,
        I_ext_fn=energy_torture_I_ext,
        reward_fn=energy_torture_reward,
    )

    res_energy_off = run_sim(
        label="Energy torture – DNA OFF",
        cfg=cfg_off_e,
        steps=steps_energy,
        I_ext_fn=energy_torture_I_ext,
        reward_fn=energy_torture_reward,
    )

    print_run_stats(res_energy_on)
    print_run_stats(res_energy_off)

    # Quick comparative summary
    print("\n" + "#" * 70)
    print("COMPARATIVE SUMMARY")
    print("#" * 70)

    def quick(res: Dict[str, Any]) -> Tuple[float, float, float, bool]:
        ATP = res["ATP"]
        risk = res["apoptosis_risk"]
        stab = res["dna_stability"]
        alive = res["alive"]
        return (
            float(np.min(ATP)),
            float(np.max(risk)),
            float(stab[len(stab) - 1]),
            bool(alive[len(alive) - 1]),
        )

    labels = [
        ("Apoptosis – DNA ON", res_apoptosis_on),
        ("Apoptosis – DNA OFF", res_apoptosis_off),
        ("Energy – DNA ON", res_energy_on),
        ("Energy – DNA OFF", res_energy_off),
    ]

    for name, res in labels:
        ATP_min, risk_max, stab_final, alive_final = quick(res)
        print(f"{name:24s} | ATP_min={ATP_min:.3f} | risk_max={risk_max:.4f} "
              f"| dna_stab_final={stab_final:.4f} | alive_final={alive_final}")


if __name__ == "__main__":
    main()



==== OCTA NEURON v9 TORTURE TESTS ====

--- Experiment 1: Apoptosis torture (chronic negative reward) ---

======================================================================
RUN: Apoptosis torture – DNA ON
======================================================================
Duration:        1000.00 ms  (steps = 20000)
Alive final:     True
Spike count:     6
V_soma range:    [-72.56, 19.66] mV
ATP min/mean:    0.829 / 0.846
Apoptosis risk:  max = 0.625519, final = 0.625519
DNA stability:   mean = 0.8538, final = 0.8561
Final summary snapshot (selected):
  alive: True
  apoptosis_risk: 0.6255190714666351
  V_soma: -57.118828897954565
  ATP: 0.8288402557373047
  homeo_rate_est: 0.8049671856681082
  recent_rate_hz: 6.0
  spike_count: 6
  dna_stability: 0.8561401152958743
  dna_error_norm: 0.8786295607662359

======================================================================
RUN: Apoptosis torture – DNA OFF
======================================================================
Duration:        1000.00 ms  (steps = 20000)
Alive final:     True
Spike count:     6
V_soma range:    [-72.59, 18.96] mV
ATP min/mean:    0.767 / 0.806
Apoptosis risk:  max = 0.625519, final = 0.625519
DNA stability:   mean = 0.8538, final = 0.8561
Final summary snapshot (selected):
  alive: True
  apoptosis_risk: 0.6255190933406042
  V_soma: -57.09254606600316
  ATP: 0.7673965096473694
  homeo_rate_est: 0.8049725523217021
  recent_rate_hz: 6.0
  spike_count: 6
  dna_stability: 0.8561401152958743
  dna_error_norm: 0.8786295607662359

--- Experiment 2: Energy (ATP) torture (high activity) ---

======================================================================
RUN: Energy torture – DNA ON
======================================================================
Duration:        1000.00 ms  (steps = 20000)
Alive final:     True
Spike count:     7
V_soma range:    [-71.01, 19.52] mV
ATP min/mean:    0.720 / 0.817
Apoptosis risk:  max = 0.000001, final = 0.000001
DNA stability:   mean = 0.8659, final = 0.8683
Final summary snapshot (selected):
  alive: True
  apoptosis_risk: 1.3041543269944818e-06
  V_soma: -52.33265758322879
  ATP: 0.8606165647506714
  homeo_rate_est: 0.9390148024821008
  recent_rate_hz: 7.0
  spike_count: 7
  dna_stability: 0.8683101220419139
  dna_error_norm: 0.7987837109238198

======================================================================
RUN: Energy torture – DNA OFF
======================================================================
Duration:        1000.00 ms  (steps = 20000)
Alive final:     True
Spike count:     7
V_soma range:    [-71.02, 16.65] mV
ATP min/mean:    0.719 / 0.771
Apoptosis risk:  max = 0.000001, final = 0.000001
DNA stability:   mean = 0.8660, final = 0.8689
Final summary snapshot (selected):
  alive: True
  apoptosis_risk: 1.3654443396614195e-06
  V_soma: -65.40333847369632
  ATP: 0.793404221534729
  homeo_rate_est: 0.9390148024821008
  recent_rate_hz: 7.0
  spike_count: 7
  dna_stability: 0.8688833256876496
  dna_error_norm: 0.7950506441187187

######################################################################
COMPARATIVE SUMMARY
######################################################################
Apoptosis – DNA ON       | ATP_min=0.829 | risk_max=0.6255 | dna_stab_final=0.8561 | alive_final=True
Apoptosis – DNA OFF      | ATP_min=0.767 | risk_max=0.6255 | dna_stab_final=0.8561 | alive_final=True
Energy – DNA ON          | ATP_min=0.720 | risk_max=0.0000 | dna_stab_final=0.8683 | alive_final=True
Energy – DNA OFF         | ATP_min=0.719 | risk_max=0.0000 | dna_stab_final=0.8689 | alive_final=True
