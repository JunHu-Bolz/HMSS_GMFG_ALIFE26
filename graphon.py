"""
Graphon kernel for the HMSS model with 4 discrete groups.

The coupling matrix W[i, j] encodes interaction strength from group j to
group i.  Groups:
  H1 (0): environment-engaged humans
  H2 (1): machine-reliant humans
  M1 (2): cooperative machines
  M2 (3): extractive machines

Z[i](t) = sum_j W[i, j] * mu_j(t)  for the simple mean-field coupling.
For human groups, Z_intra uses belief_weighted_state instead of raw mu.
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS, H1, H2, M1, M2, HUMAN_GROUPS, MACHINE_GROUPS


class Graphon:
    """4x4 discrete coupling matrix."""

    def __init__(self, cfg: HMSSConfig):
        self.cfg = cfg
        self.W = np.zeros((N_GROUPS, N_GROUPS))
        self._build_default_kernel()

    def _build_default_kernel(self):
        """Fill W with default coupling weights."""
        W = self.W

        # Human-Human block (homophily)
        W[H1, H1] = 0.5   # experts learn well from each other
        W[H1, H2] = 0.2   # experts learn less from casual users
        W[H2, H1] = 0.3   # casual users benefit from experts
        W[H2, H2] = 0.2   # casual-casual weak coupling

        # Human <- Machine (advice channels)
        W[H1, M1] = 0.3   # experts receive cooperative machine advice
        W[H1, M2] = 0.1   # experts less susceptible to extractive machines
        W[H2, M1] = 0.3   # casual users receive cooperative advice
        W[H2, M2] = 0.7   # extractive machines target casual users strongly

        # Machine <- Human (feedback channels)
        W[M1, H1] = 0.5   # cooperative machines value expert feedback
        W[M1, H2] = 0.2   # cooperative machines use casual feedback less
        W[M2, H1] = 0.3   # extractive machines exploit expert knowledge
        W[M2, H2] = 0.4   # extractive machines feed off casual users

        # Machine-Machine (coordination)
        W[M1, M1] = 0.3   # cooperative machines coordinate
        W[M1, M2] = 0.05  # cooperative avoids extractive
        W[M2, M1] = 0.1   # extractive free-rides on cooperative
        W[M2, M2] = 0.2   # extractive machines coordinate somewhat

    def compute_all_Z(self, mu, belief_weighted_state=None):
        """Compute coupling Z[i] for each group i.

        Returns Z_total, Z_intra, Z_inter each of shape (N_GROUPS,).
        Z_intra: same-species coupling (H-H or M-M).
        Z_inter: cross-species coupling (H-M or M-H).
        Z_total = Z_intra + Z_inter (for backward compat).
        """
        Z_intra = np.zeros(N_GROUPS)
        Z_inter = np.zeros(N_GROUPS)

        for i in range(N_GROUPS):
            same_species = HUMAN_GROUPS if i in HUMAN_GROUPS else MACHINE_GROUPS
            for j in range(N_GROUPS):
                # Use belief_weighted_state for human-to-human terms
                if i in HUMAN_GROUPS and j in HUMAN_GROUPS and belief_weighted_state is not None:
                    val = self.W[i, j] * belief_weighted_state[j]
                else:
                    val = self.W[i, j] * mu[j]

                if j in same_species:
                    Z_intra[i] += val
                else:
                    Z_inter[i] += val

        return Z_intra + Z_inter, Z_intra, Z_inter
