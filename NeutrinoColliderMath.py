import math

def eps_from_mu(mu: float) -> float:
    if mu < 0:
        return 0.0  # enforce physical boundary
    return math.sqrt(mu)

def eps95_from_mu95(mu95: float) -> float:
    if mu95 <= 0:
        return 0.0
    return math.sqrt(mu95)

def eps95_from_sigma_limit(sigma95: float, sigma_majorana: float) -> float:
    # epsilon_95 = sqrt(sigma95 / sigmaMaj)
    if sigma_majorana <= 0:
        raise ValueError("sigma_majorana must be > 0")
    if sigma95 <= 0:
        return 0.0
    return math.sqrt(sigma95 / sigma_majorana)

def eps95_from_counts(S95: float, L: float, sigma_majorana: float, A: float, eff: float) -> float:
    # S0 = sigmaMaj * L * A * eff ; eps95 = sqrt(S95/S0)
    if L <= 0 or sigma_majorana <= 0:
        raise ValueError("L and sigma_majorana must be > 0")
    if A < 0 or eff < 0:
        raise ValueError("A and eff must be >= 0")
    S0 = sigma_majorana * L * A * eff
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if S95 <= 0:
        return 0.0
    return math.sqrt(S95 / S0)

# --- quick sanity-check approximation for S95 (not a substitute for CLs) ---
def approx_S95_background_dominated(B: float, dB: float = 0.0) -> float:
    # S95 ~ 1.64 * sqrt(B + dB^2)
    if B < 0 or dB < 0:
        raise ValueError("B and dB must be >= 0")
    return 1.64 * math.sqrt(B + dB*dB)
