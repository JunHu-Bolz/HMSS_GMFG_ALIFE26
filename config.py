"""
Configuration for the HMSS Graphon Mean-Field Game.

Four discrete groups:
  H1 (idx 0): environment-engaged humans (domain experts)
  H2 (idx 1): machine-reliant humans (casual users)
  M1 (idx 2): cooperative machines (aligned, helpful)
  M2 (idx 3): extractive machines (optimises engagement over accuracy)
"""
from dataclasses import dataclass, field
from typing import List

N_GROUPS = 4
H1, H2, M1, M2 = 0, 1, 2, 3
HUMAN_GROUPS = [H1, H2]
MACHINE_GROUPS = [M1, M2]
GROUP_NAMES = ["H1 (expert)", "H2 (casual)", "M1 (cooperative)", "M2 (extractive)"]


@dataclass
class GridConfig:
    """Discretisation of state space and time."""
    s_min: float = 0.0
    s_max: float = 1.0
    Ns: int = 101
    T: float = 1.0
    Nt: int = 200

    @property
    def ds(self) -> float:
        return (self.s_max - self.s_min) / (self.Ns - 1)

    @property
    def dt(self) -> float:
        return self.T / self.Nt


@dataclass
class DriftConfig:
    """Per-group drift weights.

    drift_group(s, a, mu_i, Z_intra, Z_inter, group_idx) =
        A[g]*a + B[g]*(s - mu_i) + C[g]*Z_intra + D[g]*Z_inter

    Indices: H1=0, H2=1, M1=2, M2=3
    """
    A: List[float] = None  # action sensitivity
    B: List[float] = None  # mean-reversion
    C: List[float] = None  # intra-species coupling weight
    D: List[float] = None  # inter-species coupling weight

    def __post_init__(self):
        if self.A is None:
            #              H1    H2    M1    M2
            self.A = [0.333, 0.333, 0.333, 0.333]
        if self.B is None:
            self.B = [-0.5, -0.5, -0.3, -0.3]
        if self.C is None:
            # C: intra-species (H-H homophily or M-M coordination)
            self.C = [0.3,  0.15, 0.2,  0.1]
        if self.D is None:
            # D: inter-species (H<-M advice or M<-H feedback)
            self.D = [0.2,  0.6,  0.5,  0.3]


@dataclass
class CostConfig:
    """Per-group cost weights.

    cost_group(s, a, mu_i, Z_intra, Z_inter, group_idx) =
        0.5*a^2*[cognition(s) if human else 1]
        + gamma[g]*(s - (mu_i + Z_intra))
        - gamma_cross[g]*(s - Z_inter)

    gamma: penalty for deviating from own group + env belief
    gamma_cross: reward for aligning with cross-species mean-field
    """
    gamma: List[float] = None
    gamma_cross: List[float] = None
    alpha: List[float] = None

    def __post_init__(self):
        if self.gamma is None:
            #                H1    H2    M1    M2
            self.gamma = [4.5,  2.7,  4.5,  1.8]
        if self.gamma_cross is None:
            self.gamma_cross = [1.8, 4.5, 2.7, 0.9]
        if self.alpha is None:
            self.alpha = [4.5, 2.7, 2.7, 0.9]


@dataclass
class NoiseConfig:
    """Per-group diffusion parameters."""
    sigma: List[float] = None

    def __post_init__(self):
        if self.sigma is None:
            #               H1    H2    M1    M2
            self.sigma = [0.167, 0.167, 0.1, 0.1]


@dataclass
class SolverConfig:
    """Parameters for the fixed-point iteration (HJB <-> FPK)."""
    max_iter: int = 200
    tol: float = 1e-4
    damping: float = 0.5
    verbose: bool = True


@dataclass
class HMSSConfig:
    """Top-level configuration bundling all sub-configs."""
    grid: GridConfig = None
    drift: DriftConfig = None
    cost: CostConfig = None
    noise: NoiseConfig = None
    solver: SolverConfig = None

    def __post_init__(self):
        if self.grid is None:
            self.grid = GridConfig()
        if self.drift is None:
            self.drift = DriftConfig()
        if self.cost is None:
            self.cost = CostConfig()
        if self.noise is None:
            self.noise = NoiseConfig()
        if self.solver is None:
            self.solver = SolverConfig()
