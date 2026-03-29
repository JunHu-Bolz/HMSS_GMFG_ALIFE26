"""
Hamilton-Jacobi-Bellman (HJB) solver -- backward in time.

Solves for each of the 4 discrete groups independently, using
group-specific sigma, drift, and cost parameters.

Terminal condition: V(T, s) = 0.
Backward sweep: explicit Euler in time, central differences in state.
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS
from .agents import optimal_action_group, drift_group, cost_group


def _central_diff(v: np.ndarray, ds: float) -> np.ndarray:
    """Central finite difference dv/ds, one-sided at boundaries."""
    v_s = np.zeros_like(v)
    v_s[1:-1] = (v[2:] - v[:-2]) / (2.0 * ds)
    v_s[0] = (v[1] - v[0]) / ds
    v_s[-1] = (v[-1] - v[-2]) / ds
    return v_s


def _second_diff(v: np.ndarray, ds: float) -> np.ndarray:
    """Second-order central difference d2v/ds2."""
    v_ss = np.zeros_like(v)
    v_ss[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (ds ** 2)
    v_ss[0] = v_ss[1]
    v_ss[-1] = v_ss[-2]
    return v_ss


def solve_hjb(s: np.ndarray, cfg: HMSSConfig,
              mu_traj: np.ndarray, Z_intra_traj: np.ndarray,
              Z_inter_traj: np.ndarray):
    """Solve HJB backward for all 4 groups.

    Parameters
    ----------
    s : (Ns,) state grid
    cfg : HMSSConfig
    mu_traj : (Nt+1, N_GROUPS) mean-field trajectories
    Z_intra_traj : (Nt+1, N_GROUPS) intra-species coupling
    Z_inter_traj : (Nt+1, N_GROUPS) inter-species coupling

    Returns
    -------
    V_all : dict  group_idx -> (Nt+1, Ns) value function
    A_all : dict  group_idx -> (Nt+1, Ns) optimal action
    """
    Ns = cfg.grid.Ns
    Nt = cfg.grid.Nt
    ds = cfg.grid.ds
    dt = cfg.grid.dt

    V_all = {}
    A_all = {}

    for g in range(N_GROUPS):
        sigma = cfg.noise.sigma[g]
        half_sigma2 = 0.5 * sigma ** 2

        V = np.zeros((Nt + 1, Ns))
        A = np.zeros((Nt + 1, Ns))

        for n in range(Nt - 1, -1, -1):
            v = V[n + 1]
            v_s = _central_diff(v, ds)
            v_ss = _second_diff(v, ds)

            mu_n = mu_traj[n + 1, g]
            Z_intra_n = Z_intra_traj[n + 1, g]
            Z_inter_n = Z_inter_traj[n + 1, g]

            a_star = optimal_action_group(s, v_s, g, cfg)
            f = drift_group(s, a_star, mu_n, Z_intra_n, Z_inter_n, g, cfg)
            L = cost_group(s, a_star, mu_n, Z_intra_n, Z_inter_n, g, cfg)

            H_star = f * v_s + L
            V[n] = v + dt * (H_star + half_sigma2 * v_ss)
            A[n + 1] = a_star

        # Action at t=0
        v_s_0 = _central_diff(V[0], ds)
        A[0] = optimal_action_group(s, v_s_0, g, cfg)

        V_all[g] = V
        A_all[g] = A

    return V_all, A_all
