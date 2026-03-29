"""
Fokker-Planck-Kolmogorov (FPK) solver -- forward in time.

Evolves the distribution mu(t, s) for each of the 4 groups, given
optimal actions from HJB.  Different initial distributions per group.
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS, H1, H2, M1, M2
from .agents import drift_group


def _make_initial_distribution(s: np.ndarray, mean: float = 0.0,
                               std: float = 0.5) -> np.ndarray:
    """Gaussian initial distribution on the state grid."""
    mu0 = np.exp(-0.5 * ((s - mean) / std) ** 2)
    ds = s[1] - s[0]
    mu0 /= (mu0.sum() * ds)
    return mu0


# Per-group initial distribution parameters
_INIT_PARAMS = {
    H1: (0.267, 0.1),   # experts start with solid knowledge
    H2: (0.067, 0.1),   # casual users start with little knowledge
    M1: (0.200, 0.1),   # cooperative machines start helpful
    M2: (0.100, 0.1),   # extractive machines start moderate
}


def solve_fpk(s: np.ndarray, cfg: HMSSConfig,
              A_all: dict,
              mu_traj: np.ndarray, Z_intra_traj: np.ndarray,
              Z_inter_traj: np.ndarray):
    """Solve FPK forward for all 4 groups.

    Parameters
    ----------
    s : (Ns,) state grid
    cfg : HMSSConfig
    A_all : dict  group_idx -> (Nt+1, Ns) optimal action
    mu_traj : (Nt+1, N_GROUPS) mean-field trajectories (for drift eval)
    Z_intra_traj : (Nt+1, N_GROUPS) intra-species coupling
    Z_inter_traj : (Nt+1, N_GROUPS) inter-species coupling

    Returns
    -------
    density_all : dict  group_idx -> (Nt+1, Ns)
    mu_new : (Nt+1, N_GROUPS) updated mean-field trajectories
    """
    Ns = cfg.grid.Ns
    Nt = cfg.grid.Nt
    ds = cfg.grid.ds
    dt = cfg.grid.dt

    density_all = {}
    mu_new = np.zeros((Nt + 1, N_GROUPS))

    for g in range(N_GROUPS):
        sigma = cfg.noise.sigma[g]
        half_sigma2 = 0.5 * sigma ** 2

        mu0_mean, mu0_std = _INIT_PARAMS[g]

        density = np.zeros((Nt + 1, Ns))
        density[0] = _make_initial_distribution(s, mu0_mean, mu0_std)

        mf = np.zeros(Nt + 1)
        mf[0] = np.sum(s * density[0] * ds)

        for n in range(Nt):
            rho = density[n]
            mu_n = mf[n]
            Z_intra_n = Z_intra_traj[n, g]
            Z_inter_n = Z_inter_traj[n, g]
            a_n = A_all[g][n]

            f = drift_group(s, a_n, mu_n, Z_intra_n, Z_inter_n, g, cfg)

            # Flux f*rho
            flux = f * rho

            # d(f*rho)/ds central difference
            d_flux = np.zeros(Ns)
            d_flux[1:-1] = (flux[2:] - flux[:-2]) / (2.0 * ds)
            d_flux[0] = (flux[1] - flux[0]) / ds
            d_flux[-1] = (flux[-1] - flux[-2]) / ds

            # d2 rho / ds2 diffusion
            d2_rho = np.zeros(Ns)
            d2_rho[1:-1] = (rho[2:] - 2.0 * rho[1:-1] + rho[:-2]) / (ds ** 2)

            rho_new = rho + dt * (-d_flux + half_sigma2 * d2_rho)

            # Reflecting boundary at s=0: zero-flux (no probability leaks out)
            rho_new[0] = rho_new[1]

            rho_new = np.maximum(rho_new, 0.0)
            mass = rho_new.sum() * ds
            if mass > 1e-15:
                rho_new /= mass

            density[n + 1] = rho_new
            mf[n + 1] = np.sum(s * rho_new * ds)

        density_all[g] = density
        mu_new[:, g] = mf

    return density_all, mu_new
