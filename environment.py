"""
Environment model: belief dynamics and entropy for human groups.

Only human groups (H1, H2) maintain beliefs about the environment.
H1 (experts) sharpen their belief faster than H2 (casual users).
"""
import numpy as np
from .config import HMSSConfig, N_GROUPS, H1, H2, HUMAN_GROUPS


class Environment:
    """Discretised environment with belief tracking per human group."""

    def __init__(self, cfg: HMSSConfig, Ne: int = 50):
        self.cfg = cfg
        self.Ne = Ne

        self.e_grid = np.linspace(0.0, 1.0, Ne)

        # Belief b[g][j] = P(e = e_j) for each human group g
        self.belief = np.ones((N_GROUPS, Ne)) / Ne

        # True environmental state
        self.e_true = 0.5

        # Target belief: peaked at e_true
        self._b_sharp = np.exp(-50.0 * (self.e_grid - self.e_true) ** 2)
        self._b_sharp /= self._b_sharp.sum()

        # Uniform belief
        self._b_uniform = np.ones(Ne) / Ne

    def entropy(self, group_idx: int = None):
        """Discrete Shannon entropy H = -sum b_j ln(b_j).

        Returns (N_GROUPS,) or scalar if group_idx given.
        """
        b = np.clip(self.belief, 1e-15, 1.0)
        H = -np.sum(b * np.log(b), axis=1)
        if group_idx is not None:
            return H[group_idx]
        return H

    def belief_weighted_state(self, mu: np.ndarray) -> np.ndarray:
        """Belief-grounded mean-field for human groups.

        Returns mu[g] * grounding[g] for each group, where
        grounding measures how concentrated belief is near e_true.
        Machine groups return raw mu (grounding = 1).

        Parameters
        ----------
        mu : array (N_GROUPS,)

        Returns
        -------
        bws : array (N_GROUPS,)
        """
        knowledge_of_e = np.exp(-2.0 * (self.e_grid - self.e_true) ** 2)
        bws = mu.copy()
        for g in HUMAN_GROUPS:
            grounding = np.sum(self.belief[g] * knowledge_of_e)
            bws[g] = mu[g] * grounding
        return bws

    def update_belief(self, actions: np.ndarray, dt: float,
                      sharpen_scale: float = 2.0, diffuse_scale: float = 0.1):
        """Update belief for one time step based on action quality.

        Only human groups are updated. H1 sharpens faster (expert observers).

        Parameters
        ----------
        actions : array (N_GROUPS,)
            Mean action magnitude |a| per group.
        dt : float
        """
        # H1 sharpens 1.5x faster than base rate; H2 at base rate
        sharpen_multiplier = {H1: 1.5, H2: 1.0}

        for g in HUMAN_GROUPS:
            action_mag = np.abs(actions[g])
            sm = sharpen_multiplier[g]

            r_sharp = np.clip(sharpen_scale * sm * action_mag * dt, 0.0, 0.5)
            r_diffuse = np.clip(diffuse_scale * dt, 0.0, 0.5)

            self.belief[g] = (
                (1.0 - r_sharp - r_diffuse) * self.belief[g]
                + r_sharp * self._b_sharp
                + r_diffuse * self._b_uniform
            )

            self.belief[g] = np.maximum(self.belief[g], 0.0)
            total = self.belief[g].sum()
            if total > 1e-15:
                self.belief[g] /= total
