"""
Drift and cost functions for 4 discrete groups.

Groups:
  H1 (0): environment-engaged humans (domain experts)
  H2 (1): machine-reliant humans (casual users)
  M1 (2): cooperative machines
  M2 (3): extractive machines

All functions are unified by group index and vectorised over the state grid.
"""
import numpy as np
from .config import HMSSConfig, HUMAN_GROUPS


# -- Cognition cost --------------------------------------------------------

def cognition_cost(s: np.ndarray, group_idx: int) -> np.ndarray:
    """Cognition cost as a function of knowledge state.

    c(s) decreases with knowledge.  H1 (experts) have lower base cost
    than H2 (casual users).  Machines have no cognition cost (returns 1).
    Clamped to avoid overflow.
    """
    if group_idx not in HUMAN_GROUPS:
        return np.ones_like(s)
    # H1 experts: lower cognition cost (scale 0.7)
    # H2 casual:  higher cognition cost (scale 1.0)
    scale = 0.7 if group_idx == 0 else 1.0
    return scale * np.exp(-3.0 * np.clip(s, -5.0, 5.0))


# -- Drift -----------------------------------------------------------------

def drift_group(s: np.ndarray, a: np.ndarray, mu_i: float,
                Z_intra_i: float, Z_inter_i: float, group_idx: int,
                cfg: HMSSConfig) -> np.ndarray:
    """Unified drift for any group.

    f(s) = A[g]*a + B[g]*(s - mu_i) + C[g]*Z_intra + D[g]*Z_inter

    C[g] weights intra-species coupling (peer learning / coordination).
    D[g] weights inter-species coupling (machine advice / human feedback).
    """
    d = cfg.drift
    g = group_idx
    return (d.A[g] * a + d.B[g] * (s - mu_i)
            + d.C[g] * Z_intra_i + d.D[g] * Z_inter_i)


# -- Running cost ----------------------------------------------------------

def cost_group(s: np.ndarray, a: np.ndarray, mu_i: float,
               Z_intra_i: float, Z_inter_i: float, group_idx: int,
               cfg: HMSSConfig) -> np.ndarray:
    """Unified running cost for any group.

    L = 0.5 * a^2 * cognition(s, g)
        + gamma[g] * (s - mu_i)^2
        + gamma_cross[g] * (s - Z_inter)^2

    gamma penalises deviation from own group mean.
    gamma_cross penalises deviation from cross-species coupling.
    """
    c = cfg.cost
    g = group_idx
    cog = cognition_cost(s, g)
    return (0.5 * a**2 * cog
            + c.gamma[g] * (s - mu_i)**2
            + c.gamma_cross[g] * (s - Z_inter_i)**2
            - c.alpha[g] * s)


# -- Optimal action (FOC from HJB) ----------------------------------------

def optimal_action_group(s: np.ndarray, v_s: np.ndarray,
                         group_idx: int,
                         cfg: HMSSConfig) -> np.ndarray:
    """Optimal action from first-order condition.

    Human:  a* = -A[g] * v_s / cognition(s, g)
    Machine: a* = -A[g] * v_s  (cognition = 1)
    """
    g = group_idx
    cog = cognition_cost(s, g)
    a = -cfg.drift.A[g] * v_s / (cog + 1e-10)
    return np.clip(a, 0.0, 10.0)


# -- Hamiltonian under optimal action -------------------------------------

def hamiltonian_group(s: np.ndarray, v_s: np.ndarray, mu_i: float,
                      Z_intra_i: float, Z_inter_i: float, group_idx: int,
                      cfg: HMSSConfig) -> np.ndarray:
    """Optimised Hamiltonian H*(s, v_s) for one group."""
    a_star = optimal_action_group(s, v_s, group_idx, cfg)
    f = drift_group(s, a_star, mu_i, Z_intra_i, Z_inter_i, group_idx, cfg)
    L = cost_group(s, a_star, mu_i, Z_intra_i, Z_inter_i, group_idx, cfg)
    return f * v_s + L
