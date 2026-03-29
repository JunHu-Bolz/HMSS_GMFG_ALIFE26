"""
Information flow metrics and parasitism indicators for 4-group model.

Groups: H1(0), H2(1), M1(2), M2(3).
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS, H1, H2, M1, M2, HUMAN_GROUPS, MACHINE_GROUPS


def _smooth(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving average to smooth noisy finite differences."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    # pad edges to preserve length
    padded = np.pad(x, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(x)]


def info_env_to_human(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Information flow from environment to humans: -dH/dt.

    Averaged over human groups H1 and H2, smoothed to remove
    finite-difference noise.
    Returns (Nt,).
    """
    entropy = results['entropy'][:, HUMAN_GROUPS]  # (Nt+1, 2)
    mean_entropy = entropy.mean(axis=1)
    dt = cfg.grid.dt
    raw = -(mean_entropy[1:] - mean_entropy[:-1]) / dt
    return _smooth(raw)


def info_env_to_group(results: dict, cfg: HMSSConfig, group_idx: int) -> np.ndarray:
    """Information flow from environment to a specific human group: -dH/dt.

    Returns (Nt,).
    """
    entropy = results['entropy'][:, group_idx]
    dt = cfg.grid.dt
    raw = -(entropy[1:] - entropy[:-1]) / dt
    return _smooth(raw)


def info_human_to_machine(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Information flow from humans to machines: mean Z over machine groups.

    Returns (Nt+1,).
    """
    Z = results['Z'][:, MACHINE_GROUPS]
    return Z.mean(axis=1)


def info_machine_to_human(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Information flow from machines to humans: mean Z over human groups.

    Returns (Nt+1,).
    """
    Z = results['Z'][:, HUMAN_GROUPS]
    return Z.mean(axis=1)


def reliance_ratio(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Human reliance on machines vs environment-grounded peers.

    For each human group, ratio of machine-coupling to total coupling.
    Averaged over H1 and H2.

    Returns (Nt+1,).
    """
    graphon = results['graphon']
    W = graphon.W
    mu = results['mu']  # (Nt+1, N_GROUPS)

    Nt = cfg.grid.Nt
    R = np.zeros(Nt + 1)

    for n in range(Nt + 1):
        r_vals = []
        for h in HUMAN_GROUPS:
            # Machine coupling contribution
            machine_contrib = sum(W[h, m] * np.abs(mu[n, m]) for m in MACHINE_GROUPS)
            # Human coupling contribution
            human_contrib = sum(W[h, hh] * np.abs(mu[n, hh]) for hh in HUMAN_GROUPS)
            total = machine_contrib + human_contrib
            if total > 1e-10:
                r_vals.append(machine_contrib / total)
            else:
                r_vals.append(0.0)
        R[n] = np.mean(r_vals)

    return np.clip(R, 0.0, 1.0)


def reliance_ratio_per_group(results: dict, cfg: HMSSConfig,
                             group_idx: int) -> np.ndarray:
    """Reliance ratio for a specific human group. Returns (Nt+1,)."""
    graphon = results['graphon']
    W = graphon.W
    mu = results['mu']
    Nt = cfg.grid.Nt
    R = np.zeros(Nt + 1)

    for n in range(Nt + 1):
        machine_contrib = sum(W[group_idx, m] * np.abs(mu[n, m])
                              for m in MACHINE_GROUPS)
        human_contrib = sum(W[group_idx, h] * np.abs(mu[n, h])
                            for h in HUMAN_GROUPS)
        total = machine_contrib + human_contrib
        R[n] = machine_contrib / total if total > 1e-10 else 0.0

    return np.clip(R, 0.0, 1.0)


def human_action_magnitude(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Mean action magnitude of human agents over time. Returns (Nt+1,)."""
    return results['mean_actions'][:, HUMAN_GROUPS].mean(axis=1)


def human_uncertainty(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Human uncertainty: L2 distance from current belief to target b*.

    Averaged over human groups, normalised to [0, 1] by dividing by
    the maximum possible distance (uniform belief to b*).

    Returns (Nt+1,).
    """
    belief_history = results['belief_history']  # (Nt+1, N_GROUPS, Ne)
    env = results['env']
    b_star = env._b_sharp                       # (Ne,)
    b_uniform = env._b_uniform                  # (Ne,)

    # Max possible L2 distance (uniform -> b*)
    max_dist = np.linalg.norm(b_uniform - b_star)

    Nt = cfg.grid.Nt
    hu = np.zeros(Nt + 1)
    for n in range(Nt + 1):
        dists = [np.linalg.norm(belief_history[n, g] - b_star)
                 for g in HUMAN_GROUPS]
        hu[n] = np.mean(dists) / max(max_dist, 1e-10)

    return np.clip(hu, 0.0, 1.0)


def parasitism_index(results: dict, cfg: HMSSConfig) -> np.ndarray:
    """Composite parasitism index P(t), equally weighted sum.

    P = (1/2) * reliance + (1/2) * extraction
    Extraction uses a fixed scale (Z_SCALE=2.0) so that values are
    comparable across scenarios.
    Returns (Nt+1,).
    """
    Z_SCALE = 0.667

    R = reliance_ratio(results, cfg)

    # Extraction: human influence on machines, fixed-scale normalisation
    Z_m = results['Z_inter'][:, MACHINE_GROUPS].mean(axis=1)
    extraction = np.clip(np.abs(Z_m) / Z_SCALE, 0.0, 1.0)

    return (R + extraction) / 2.0


def counterfactual_info_flow(results: dict, cfg: HMSSConfig) -> dict:
    """Counterfactual (perturbation-based) information flow.

    For each directed channel, re-run the solver with that channel's
    graphon weights zeroed out. The causal information flow is measured
    as the difference in mean-field trajectories:

        IF(X -> Y)(t) = |mu_Y(t) - mu_Y_no_X(t)|

    This directly answers: "how much would Y's trajectory change if
    we removed X's influence?"

    Returns dict with keys:
        'human_to_machine' : (Nt+1,)  -- IF(H -> M)
        'machine_to_human' : (Nt+1,)  -- IF(M -> H)
    """
    from .mfg_solver import solve
    import copy

    # --- Counterfactual 1: remove human -> machine coupling ---
    W_no_h2m = {}
    for h in HUMAN_GROUPS:
        for m in MACHINE_GROUPS:
            W_no_h2m[(h, m)] = results['graphon'].W[h, m]  # keep H<-M
    # Zero out M<-H weights
    W_zero_h2m = {}
    for m in MACHINE_GROUPS:
        for h in HUMAN_GROUPS:
            gname_m = ['H1', 'H2', 'M1', 'M2'][m]
            gname_h = ['H1', 'H2', 'M1', 'M2'][h]
            W_zero_h2m[(gname_m, gname_h)] = 0.0

    cfg_cf1 = copy.deepcopy(cfg)
    cfg_cf1.solver.verbose = False
    cfg_cf1.solver.tol = min(cfg.solver.tol, 1e-5)
    res_no_h2m = solve(cfg_cf1, W_overrides=W_zero_h2m)

    # --- Counterfactual 2: remove machine -> human coupling ---
    W_zero_m2h = {}
    for h in HUMAN_GROUPS:
        for m in MACHINE_GROUPS:
            gname_h = ['H1', 'H2', 'M1', 'M2'][h]
            gname_m = ['H1', 'H2', 'M1', 'M2'][m]
            W_zero_m2h[(gname_h, gname_m)] = 0.0

    cfg_cf2 = copy.deepcopy(cfg)
    cfg_cf2.solver.verbose = False
    cfg_cf2.solver.tol = min(cfg.solver.tol, 1e-5)
    res_no_m2h = solve(cfg_cf2, W_overrides=W_zero_m2h)

    # --- Compute causal impact ---
    mu_full = results['mu']

    # H->M: difference in machine trajectories when H->M is removed
    mu_m_full = mu_full[:, MACHINE_GROUPS].mean(axis=1)
    mu_m_no_h = res_no_h2m['mu'][:, MACHINE_GROUPS].mean(axis=1)
    if_h2m = _smooth(np.abs(mu_m_full - mu_m_no_h), window=15)

    # M->H: difference in human trajectories when M->H is removed
    mu_h_full = mu_full[:, HUMAN_GROUPS].mean(axis=1)
    mu_h_no_m = res_no_m2h['mu'][:, HUMAN_GROUPS].mean(axis=1)
    if_m2h = _smooth(np.abs(mu_h_full - mu_h_no_m), window=15)

    return {
        'human_to_machine': if_h2m,
        'machine_to_human': if_m2h,
        'mu_m_no_h': res_no_h2m['mu'],
        'mu_h_no_m': res_no_m2h['mu'],
    }


def summary(results: dict, cfg: HMSSConfig) -> dict:
    """Compute all metrics and return as a dict."""
    mu = results['mu']
    cf = counterfactual_info_flow(results, cfg)
    return {
        'info_env_to_human': info_env_to_human(results, cfg),
        'info_human_to_machine': info_human_to_machine(results, cfg),
        'info_machine_to_human': info_machine_to_human(results, cfg),
        'cf_human_to_machine': cf['human_to_machine'],
        'cf_machine_to_human': cf['machine_to_human'],
        'reliance_ratio': reliance_ratio(results, cfg),
        'action_magnitude': human_action_magnitude(results, cfg),
        'parasitism_index': parasitism_index(results, cfg),
        'human_uncertainty': human_uncertainty(results, cfg),
        'entropy_mean': results['entropy'][:, HUMAN_GROUPS].mean(axis=1),
        'mu_H1': mu[:, H1],
        'mu_H2': mu[:, H2],
        'mu_M1': mu[:, M1],
        'mu_M2': mu[:, M2],
        'mu_h_mean': mu[:, HUMAN_GROUPS].mean(axis=1),
        'mu_m_mean': mu[:, MACHINE_GROUPS].mean(axis=1),
    }
