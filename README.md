Graphon Mean-Field Game for Human-Machine Social System
A graphon mean-field game (GMFG) model of how knowledge and influence flow through a population of interacting humans and machines, and of when machine influence becomes parasitic rather than mutualistic.

Code for Hu-Bolz & Stovold, ALife 2026.

Overview
The model studies a Human-Machine Social System (HMSS) at the population scale. Agents are partitioned into four groups that interact through a weighted coupling kernel (the graphon). Each agent holds a scalar state in [0, 1] representing its grounding in the underlying environment (roughly, how well-calibrated its knowledge is). Agents choose actions to balance the effort of moving their state against the cost of being misaligned with their own group and with the groups they are coupled to.

The four groups are:
 
| Index | Group | Interpretation |
|-------|-------|----------------|
| 0 | H1 | Environment-engaged humans (domain experts) |
| 1 | H2 | Machine-reliant humans (casual users) |
| 2 | M1 | Cooperative machines (aligned, helpful) |
| 3 | M2 | Extractive machines (optimise engagement over accuracy) |

The population equilibrium is the fixed point of two coupled PDEs solved over each group:
 
1. A Hamilton-Jacobi-Bellman (HJB) equation, integrated backward in time, giving the optimal value function and feedback control.
2. A Fokker-Planck-Kolmogorov (FPK) equation, integrated forward in time, giving the resulting density and the group mean-field trajectory.
These are iterated to convergence with damping. Alongside the agent dynamics, an environment object maintains a belief distribution per human group; its entropy tracks how well humans stay grounded, and its rate of change defines the information flow from the environment into the human population.
 
From the converged solution, the code derives information-flow and parasitism metrics, including a composite parasitism index and counterfactual (channel-ablation) measures of causal influence between humans and machines.
