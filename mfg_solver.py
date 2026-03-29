"""
GMFG fixed-point solver: iterates HJB <-> FPK until convergence.

mu is (Nt+1, N_GROUPS) -- one mean-field scalar per group per time step.
Z is (Nt+1, N_GROUPS) -- coupling Z[i] = sum_j W[i,j] * mu_j(t).
For human groups, Z uses belief_weighted_state for the H-H terms.
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS, HUMAN_GROUPS
from .graphon import Graphon
from .environment import Environment
from .hjb_solver import solve_hjb
from .fpk_solver import solve_fpk


def _compute_Z_trajectories(graphon: Graphon, env: Environment,
                            mu_traj: np.ndarray, cfg: HMSSConfig):
    """Compute coupling Z trajectories for all time steps.

    Returns Z, Z_intra, Z_inter each of shape (Nt+1, N_GROUPS).
    """
    Nt = cfg.grid.Nt
    Z = np.zeros((Nt + 1, N_GROUPS))
    Z_intra = np.zeros((Nt + 1, N_GROUPS))
    Z_inter = np.zeros((Nt + 1, N_GROUPS))

    for n in range(Nt + 1):
        bws = env.belief_weighted_state(mu_traj[n])
        Z[n], Z_intra[n], Z_inter[n] = graphon.compute_all_Z(mu_traj[n], bws)

    return Z, Z_intra, Z_inter


def _compute_mean_actions(A_all, density_all, cfg):
    """Compute distribution-weighted mean action per group per time step.

    Returns mean_actions (Nt+1, N_GROUPS).
    """
    Nt = cfg.grid.Nt
    ds = cfg.grid.ds
    mean_actions = np.zeros((Nt + 1, N_GROUPS))

    for g in range(N_GROUPS):
        if g in A_all and g in density_all:
            for n in range(Nt + 1):
                mean_actions[n, g] = np.sum(
                    np.abs(A_all[g][n]) * density_all[g][n] * ds
                )
    return mean_actions


def _compute_entropy_trajectory(cfg, mean_actions):
    """Evolve environment belief forward and record entropy, grounding, belief.

    Returns entropy (Nt+1, N_GROUPS), grounding (Nt+1, N_GROUPS),
            belief_history (Nt+1, N_GROUPS, Ne).
    """
    Nt = cfg.grid.Nt
    env_rec = Environment(cfg)
    Ne = env_rec.Ne
    entropy = np.zeros((Nt + 1, N_GROUPS))
    grounding = np.zeros((Nt + 1, N_GROUPS))
    belief_history = np.zeros((Nt + 1, N_GROUPS, Ne))
    knowledge_of_e = np.exp(-2.0 * (env_rec.e_grid - env_rec.e_true) ** 2)

    entropy[0] = env_rec.entropy()
    belief_history[0] = env_rec.belief.copy()
    for g in HUMAN_GROUPS:
        grounding[0, g] = np.sum(env_rec.belief[g] * knowledge_of_e)

    for n in range(Nt):
        env_rec.update_belief(mean_actions[n], cfg.grid.dt)
        entropy[n + 1] = env_rec.entropy()
        belief_history[n + 1] = env_rec.belief.copy()
        for g in HUMAN_GROUPS:
            grounding[n + 1, g] = np.sum(env_rec.belief[g] * knowledge_of_e)

    return entropy, grounding, belief_history


def solve(cfg: HMSSConfig = None, W_overrides: dict = None):
    """Run the GMFG fixed-point iteration.

    Parameters
    ----------
    cfg : HMSSConfig or None
    W_overrides : dict or None
        Optional overrides for graphon entries, e.g.
        {('H2','M2'): 0.9} sets W[H2, M2] = 0.9.

    Returns
    -------
    results : dict with keys:
        'V'            : dict group_idx -> (Nt+1, Ns)
        'A'            : dict group_idx -> (Nt+1, Ns)
        'density'      : dict group_idx -> (Nt+1, Ns)
        'mu'           : (Nt+1, N_GROUPS) mean-field trajectories
        'entropy'      : (Nt+1, N_GROUPS) environment entropy
        'belief_history': (Nt+1, N_GROUPS, Ne) belief distributions over time
        'Z'            : (Nt+1, N_GROUPS) coupling trajectories
        'mean_actions' : (Nt+1, N_GROUPS)
        'graphon'      : Graphon instance
        'env'          : Environment instance
        's'            : (Ns,) state grid
        'converged'    : bool
        'iterations'   : int
    """
    if cfg is None:
        cfg = HMSSConfig()

    graphon = Graphon(cfg)
    if W_overrides:
        name_to_idx = {'H1': 0, 'H2': 1, 'M1': 2, 'M2': 3}
        for (ni, nj), val in W_overrides.items():
            i = name_to_idx.get(ni, ni)
            j = name_to_idx.get(nj, nj)
            graphon.W[i, j] = val
    env = Environment(cfg)

    Ns = cfg.grid.Ns
    Nt = cfg.grid.Nt
    s = np.linspace(cfg.grid.s_min, cfg.grid.s_max, Ns)

    # Initialise mean-field trajectories (match FPK init means)
    mu = np.zeros((Nt + 1, N_GROUPS))
    mu[:, 0] = 0.267  # H1: experts start with solid knowledge
    mu[:, 1] = 0.067  # H2: casual users start with little knowledge
    mu[:, 2] = 0.200  # M1: cooperative machines start helpful
    mu[:, 3] = 0.100  # M2: extractive machines start moderate

    converged = False
    iteration = 0
    err = float('inf')

    for iteration in range(cfg.solver.max_iter):
        # Compute coupling (split into intra/inter species)
        Z, Z_intra, Z_inter = _compute_Z_trajectories(graphon, env, mu, cfg)

        # HJB backward
        V_all, A_all = solve_hjb(s, cfg, mu, Z_intra, Z_inter)

        # FPK forward
        density_all, mu_new = solve_fpk(s, cfg, A_all, mu, Z_intra, Z_inter)

        # Convergence check
        err = np.max(np.abs(mu_new - mu))

        if cfg.solver.verbose:
            errs = [np.max(np.abs(mu_new[:, g] - mu[:, g]))
                    for g in range(N_GROUPS)]
            print(f"Iter {iteration:3d}  |  "
                  f"err_H1={errs[0]:.6f}  err_H2={errs[1]:.6f}  "
                  f"err_M1={errs[2]:.6f}  err_M2={errs[3]:.6f}  "
                  f"err={err:.6f}")

        if err < cfg.solver.tol:
            converged = True
            mu = mu_new
            if cfg.solver.verbose:
                print(f"Converged after {iteration + 1} iterations.")
            break

        # Damped update
        alpha = cfg.solver.damping
        mu = alpha * mu + (1.0 - alpha) * mu_new

        # Update environment
        mean_actions = _compute_mean_actions(A_all, density_all, cfg)
        env_copy = Environment(cfg)
        for n in range(Nt):
            env_copy.update_belief(mean_actions[n], cfg.grid.dt)
        env = env_copy

    else:
        if cfg.solver.verbose:
            print(f"Did not converge after {cfg.solver.max_iter} iterations "
                  f"(err={err:.6f}).")

    # Final recompute
    Z, Z_intra, Z_inter = _compute_Z_trajectories(graphon, env, mu, cfg)
    mean_actions = _compute_mean_actions(A_all, density_all, cfg)
    entropy_history, grounding_history, belief_history = _compute_entropy_trajectory(cfg, mean_actions)

    return {
        'V': V_all,
        'A': A_all,
        'density': density_all,
        'mu': mu,
        'entropy': entropy_history,
        'grounding': grounding_history,
        'belief_history': belief_history,
        'Z': Z,
        'Z_intra': Z_intra,
        'Z_inter': Z_inter,
        'mean_actions': mean_actions,
        'graphon': graphon,
        'env': env,
        's': s,
        'converged': converged,
        'iterations': iteration + 1,
    }
