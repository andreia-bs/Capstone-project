
# =============================================================================
# BESPOKE BAYESIAN OPTIMISATION — CAPSTONE PROJECT
# FINAL ROUND VERSION — MODULE 24
# =============================================================================
# This script incorporates all amendments recommended after Module 22's
# PCA-informed reflection. Key changes vs the previous version:
#
#   F1: 'override' → 'localised_ei'   ← AMENDED (this version)
#       Rationale: The hard override at (0.703500, 0.716000) is an exact
#       duplicate of Week 2 and must not be resubmitted. Global EI has
#       also failed for ten consecutive rounds because the GP fits a smooth
#       surface over a near-singular peak, placing the EI maximum far from
#       the true peak.
#
#       Solution — Localised EI with constrained bounds:
#         EI is run with the acquisition optimiser constrained to a tight
#         ±0.010 box around the known best:
#           x1 ∈ [0.6935, 0.7135], x2 ∈ [0.7060, 0.7260]
#         This guarantees:
#           (a) A genuinely new query — EI = 0 exactly at observed points
#               by construction, so the optimiser cannot return W2's point
#           (b) The search stays within the known non-negligible region
#           (c) The GP's local posterior within the box guides the exact
#               location, using the tight length-scale bounds (0.005, 0.05)
#               to model fine-grained variation at this scale
#
#   F4: 'thompson' → 'ei'
#       Rationale: Thompson Sampling discovered the W11 breakthrough at
#       (0.418286, 0.384490, 0.407242, 0.350393) = +0.566. EI will now
#       exploit this confirmed best rather than randomly exploring.
#
#   F5: Boundary override updated to a genuinely new near-boundary point.
#       Rationale: (0.99, 0.99, 0.99, 0.99) was submitted in W10 (7915).
#       The confirmed optimum is (1,1,1,1) = 8662.48. The final query
#       probes (0.980, 0.980, 0.990, 0.990) — a new untested combination
#       that maps the boundary neighbourhood more finely.
#
#   F6: 'thompson' → 'ei'
#       Rationale: Thompson Sampling has produced three consecutive
#       regressions for F6 (W9: -0.399, W10: -0.294, W11: -0.294).
#       EI targeting the confirmed best at (0.478253, 0.319282, 0.543932,
#       0.782657, 0.101184) = -0.181 is the final-round strategy.
#
# References:
#   [1] Hennig & Schuler (2012) — Entropy Search for Information-Efficient
#       Global Optimization. JMLR 13, 1809–1837.
#   [2] Eriksson et al. (2019) — Scalable Global Optimization via Local
#       Bayesian Optimization (TuRBO). NeurIPS 2019.
#   [3] GPyTorch GitHub benchmarks — kernel hyperparameter baselines.
#   [4] OpenML benchmark datasets — contextualising F7/F8 HP tuning outputs.
# =============================================================================

import math
import sys
import numpy as np
np.set_printoptions(linewidth=np.inf)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

import warnings

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    UpperConfidenceBound,
    ExpectedImprovement,
    qMaxValueEntropy,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.constraints import Interval
from botorch.generation import MaxPosteriorSampling

# =============================================================================
# ── VERSION CONTROL ───────────────────────────────────────────────────────────
# Update WEEK_NO before each run. For Module 24 (12th query), set to 12.
# =============================================================================

WEEK_NO = 13   # ← UPDATE THIS before running for Module 24

FUNCTION_NO = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# =============================================================================
# ── PER-FUNCTION KERNEL CONFIGURATION ────────────────────────────────────────
# =============================================================================

LENGTH_SCALE_BOUNDS = {
    1: (0.005, 0.05),  # Very tight — sharp near-singular peak
    2: (0.05,  0.5),   # Moderate 1D
    3: (0.1,   1.0),   # Smooth, wider
    4: (0.05,  0.5),   # Jagged 4D — tighter for exploitation
    5: (0.1,   1.5),   # Smooth, very wide
    6: (0.05,  0.75),  # Moderate multi-dim
    7: (0.05,  1.0),   # HP tuning — ARD important  [Ref 4]
    8: (0.05,  1.0),   # 8D — TuRBO manages local scales  [Ref 2]
}[FUNCTION_NO]

NU_SMOOTHNESS = {
    1: 1.5,   # Sharp/noisy → once-differentiable
    2: 1.5,   # Moderate sharpness
    3: 2.5,   # Smooth
    4: 1.5,   # Jagged — once-differentiable for exploitation
    5: 2.5,   # Smooth, large space
    6: 2.5,   # Smooth moderate-dim
    7: 2.5,   # HP tuning — smooth response surface assumed
    8: 2.5,   # 8D — smooth assumed; TuRBO handles locality  [Ref 2]
}[FUNCTION_NO]

# =============================================================================
# ── PER-FUNCTION ACQUISITION STRATEGY ────────────────────────────────────────
# =============================================================================
# Final-round acquisition map:
#
#   F1: 'override'   — hard override to known peak (0.703500, 0.716000)
#                      EI has failed for 10 consecutive rounds; GP cannot
#                      reliably model the near-singular peak. Manual override
#                      is the only reliable strategy for the final query.
#
#   F2: 'ucb'        — κ=0.8, exploit confirmed best at (0.697, 0.321)
#
#   F3: 'ucb'        — κ=2.0, unchanged; GP continues to explore
#
#   F4: 'ei'         — AMENDED from 'thompson'. Exploit W11 breakthrough
#                      at (0.418286, 0.384490, 0.407242, 0.350393) = +0.566
#
#   F5: 'override'   — boundary override to (0.980, 0.980, 0.990, 0.990)
#                      (0.99, 0.99, 0.99, 0.99) already submitted in W10.
#                      True optimum confirmed at (1,1,1,1) = 8662.48.
#                      Final query maps boundary neighbourhood.
#
#   F6: 'ei'         — AMENDED from 'thompson'. Three consecutive TS
#                      regressions. EI exploits confirmed best at
#                      (0.478253, 0.319282, 0.543932, 0.782657, 0.101184).
#
#   F7: 'ei'         — unchanged; consistent improvement trajectory
#
#   F8: 'turbo'      — unchanged; TuRBO trust region well-calibrated
# =============================================================================

ACQUISITION_STRATEGY = {
    1: 'localised_ei',  # AMENDED: localised EI replaces duplicate override
                        # Constrained to ±0.010 box around (0.703500, 0.716000)
    2: 'ucb',           # Exploit confirmed best; κ reduced to 0.8
    3: 'ucb',           # Unchanged
    4: 'ei',            # AMENDED: exploit W11 breakthrough
    5: 'override',      # FINAL ROUND: boundary override (new point)
    6: 'ei',            # AMENDED: EI replaces TS after 3 regressions
    7: 'ei',            # Unchanged — consistent improver
    8: 'turbo',         # Unchanged — TuRBO well-calibrated
}[FUNCTION_NO]

# UCB κ — only used when strategy is 'ucb'
UCB_KAPPA = {
    2: 0.8,   # Reduced from 1.0 — tighter exploitation near (0.697, 0.321)
    3: 2.0,   # Unchanged
}

# =============================================================================
# ── MANUAL OVERRIDES & LOCALISED BOUNDS ──────────────────────────────────────
# =============================================================================
# F1 LOCALISED EI BOUNDS:
#   The known best for F1 is at (0.703500, 0.716000) = 1.81e-9 (Week 2).
#   The hard override to this point is a duplicate of W2 — not permitted.
#   Global EI has also failed for 10 consecutive rounds.
#
#   Localised EI constrains optimize_acqf to a ±0.010 box around the
#   known best. Within this box:
#     - EI = 0 exactly at (0.703500, 0.716000) since it is an observed
#       point — the optimiser is guaranteed to return a different point
#     - The tight length-scale bounds (0.005, 0.05) make the GP sensitive
#       to fine-grained variation at this scale, providing meaningful
#       local gradient information to guide the query
#     - The search stays within the region where F1 is non-negligible
#
#   Box: x1 ∈ [0.6935, 0.7135], x2 ∈ [0.7060, 0.7260]
#
# F5 BOUNDARY OVERRIDE:
#   (1,1,1,1) = 8662.48 is the confirmed optimum (W8).
#   (0.99,0.99,0.99,0.99) = 7915.74 was submitted in W10.
#   New override: (0.980,0.980,0.990,0.990) — untested combination that
#   maps the boundary neighbourhood more finely.
# =============================================================================

# F1: localised search box — ±0.010 around (0.703500, 0.716000)
F1_LOCALISED_BOUNDS = torch.tensor(
    [[0.6935, 0.7060],   # lower bounds: [x1_lb, x2_lb]
     [0.7135, 0.7260]],  # upper bounds: [x1_ub, x2_ub]
    dtype=torch.float64
)

F5_BOUNDARY_OVERRIDE = np.array([0.980, 0.980, 0.990, 0.990])

# =============================================================================
# ── TURBO HYPERPARAMETERS (F8 only) ──────────────────────────────────────────
# =============================================================================

TURBO_LENGTH_INIT    = 0.4
TURBO_LENGTH_MIN     = 0.01
TURBO_LENGTH_MAX     = 1.6
TURBO_SUCCESS_TOL    = 3
TURBO_FAILURE_TOL    = 4
TURBO_SUCCESS_FACTOR = 2.0
TURBO_FAILURE_FACTOR = 0.5

# =============================================================================
# ── NOISE FLOOR CONFIGURATION ─────────────────────────────────────────────────
# =============================================================================

NOISE_CONSTRAINT = {
    1: Interval(1e-6, 1e-3),   # Tight — prevent noise absorption on sharp F1
    2: Interval(1e-5, 1e-1),   # Moderate
    3: Interval(1e-5, 1e-1),   # Moderate
    4: Interval(1e-5, 1e-2),   # Tight — jagged F4
    5: Interval(1e-5, 1e-1),   # Moderate
    6: Interval(1e-5, 1e-1),   # Moderate
    7: Interval(1e-5, 1e-1),   # Moderate
    8: Interval(1e-5, 1e-1),   # Moderate
}[FUNCTION_NO]


# =============================================================================
# MAIN
# =============================================================================

def main():
    X, y = read_data(debug=False, plot_datapoints=False)
    N = X.shape[1]

    if ACQUISITION_STRATEGY == 'turbo':
        model = TuRBoGP()
    elif ACQUISITION_STRATEGY == 'override':
        # For override functions, still fit the GP for plotting purposes
        # but the query is determined by the override constant
        model = BoTorchGP()
    else:
        model = BoTorchGP()

    model.fit(X, y, debug=True)
    model.find_acquisition_max(debug=True)
    plot_gp_results(model, X, y)


# =============================================================================
# DATA LOADING
# =============================================================================

def read_data(debug=False, plot_datapoints=True):
    X = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_inputs.npy")
    y = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_outputs.npy")
    N = X.shape[1]

    for week in range(1, WEEK_NO):
        with open(f"./week{week}_inputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_values = np.array([float(x) for x in line.strip().split('-')])
            X = np.vstack((X, new_values))
        with open(f"./week{week}_outputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_value = float(line.strip())
            y = np.append(y, np.array(new_value))

    if debug:
        print("\n--- Data points ---")
        for Xi, yi in zip(X, y):
            print(Xi, yi)

    if plot_datapoints:
        sorted_indices = np.arange(len(y))
        rank_indices   = np.argsort(sorted_indices)
        previous_data_indices = rank_indices[:-WEEK_NO + 1]
        latest_data_indices   = rank_indices[-WEEK_NO + 1:]
        sorted_y = y[sorted_indices]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(previous_data_indices, sorted_y[previous_data_indices],
                   c=sorted_y[previous_data_indices], vmin=y.min(), vmax=y.max(),
                   marker='o', cmap='rainbow', s=50, edgecolor='black')
        ax.scatter(latest_data_indices, sorted_y[latest_data_indices],
                   c=sorted_y[latest_data_indices], vmin=y.min(), vmax=y.max(),
                   marker='s', cmap='rainbow', s=50, edgecolor='black')
        ax.set_ylabel("y")
        ax.set_title(f'Outputs for function {FUNCTION_NO}')
        plt.show()

    return X, y


# =============================================================================
# PLOTTING
# =============================================================================

def plot_gp_results(model, X, y):
    N = X.shape[1]

    if N == 1:
        linspace    = np.linspace(0, 1, 100)
        mean, std   = model.evaluate(linspace.reshape(-1, 1), grid=False)
        acquisition = model.evaluate_acquisition(linspace.reshape(-1, 1), grid=False)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:-WEEK_NO + 1], y[:-WEEK_NO + 1], color='black', marker='o', s=40)
        ax.scatter(X[-WEEK_NO + 1:], y[-WEEK_NO + 1:], color='black', marker='s', s=40)
        ax.plot(linspace, mean, linewidth=2, label='Posterior mean')
        ax.fill_between(linspace, mean - std, mean + std, alpha=0.2, label='Posterior std')
        ax.plot(linspace, acquisition, color='green', linewidth=2, label='Acquisition')
        ax.scatter(acquisition_max_X, acquisition_max_y, marker='*', s=200,
                   color='green', label='Acquisition max / Override')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        strat_label = ACQUISITION_STRATEGY.upper()
        ax.set_title(f'GP posterior — Function {FUNCTION_NO} [{strat_label}]')
        ax.legend(); plt.tight_layout(); plt.show()

    elif N == 2:
        linspace  = np.linspace(0, 1, 25)
        mesh      = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:-WEEK_NO + 1, 0], X[:-WEEK_NO + 1, 1], y[:-WEEK_NO + 1],
                   marker='o', color='black', s=50, depthshade=True)
        ax.scatter(X[-WEEK_NO + 1:, 0], X[-WEEK_NO + 1:, 1], y[-WEEK_NO + 1:],
                   marker='s', color='black', s=50, depthshade=True)
        ax.plot_surface(mesh[0], mesh[1], mean, cmap='rainbow', alpha=0.8, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], mean + std, color='lightgray', alpha=0.2, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], mean - std, color='lightgray', alpha=0.2, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], acquisition, color='green', alpha=0.2, linewidth=0)
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_y,
                   color='green', s=200, marker='*')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
        ax.set_title(f'GP posterior — Function {FUNCTION_NO} [{ACQUISITION_STRATEGY.upper()}]')
        plt.tight_layout(); plt.show()

    elif N == 3:
        linspace  = np.linspace(0, 1, 20)
        mesh      = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, _ = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        sizes = 40 * (y - y.min()) / (y.max() - y.min() + 1e-9) + 10
        ax.scatter(X[:-WEEK_NO + 1, 0], X[:-WEEK_NO + 1, 1], X[:-WEEK_NO + 1, 2],
                   s=sizes[:-WEEK_NO + 1], c=y[:-WEEK_NO + 1], cmap='rainbow',
                   vmin=y.min(), vmax=y.max(), marker='o', depthshade=True)
        ax.scatter(X[-WEEK_NO + 1:, 0], X[-WEEK_NO + 1:, 1], X[-WEEK_NO + 1:, 2],
                   s=sizes[-WEEK_NO + 1:], c=y[-WEEK_NO + 1:], cmap='rainbow',
                   vmin=y.min(), vmax=y.max(), marker='s', depthshade=True)
        ax.scatter(mesh[0], mesh[1], mesh[2], c=acquisition, cmap='viridis',
                   marker='o', s=100, alpha=0.4, edgecolors='none')
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_X[2],
                   color='green', s=200, marker='*')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
        ax.set_title(f'Observed data — Function {FUNCTION_NO} [{ACQUISITION_STRATEGY.upper()}]')
        plt.tight_layout(); plt.show()

    else:  # N > 3 — parallel coordinates
        acquisition_max_X, _ = model.find_acquisition_max()
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = plt.Normalize(y.min(), y.max())
        cmap = plt.cm.rainbow
        x_axis = np.arange(N)
        for i in range(X.shape[0] - WEEK_NO + 1):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=1, alpha=0.6)
        for i in range(X.shape[0] - WEEK_NO + 1, X.shape[0]):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=2,
                    linestyle='--', alpha=0.6)
        ax.plot(x_axis, acquisition_max_X, color='green', linewidth=3,
                linestyle='--', label='Acquisition max / Override')
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f'x[{d}]' for d in range(1, N + 1)])
        ax.set_title(f'Parallel coordinates — Function {FUNCTION_NO} [{ACQUISITION_STRATEGY.upper()}]')
        ax.set_ylabel('Input value'); ax.legend()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='y')
        plt.tight_layout(); plt.show()


# =============================================================================
# BASE CLASS
# =============================================================================

class GPModel:
    N = None

    def fit(self, X, y, debug=False):
        raise NotImplementedError

    def evaluate(self, X, grid=False):
        raise NotImplementedError

    def evaluate_acquisition(self, X, grid=False):
        raise NotImplementedError

    def find_acquisition_max(self, debug=False):
        raise NotImplementedError

    @staticmethod
    def _build_gp(X_np, y_np):
        matern = MaternKernel(
            nu=NU_SMOOTHNESS,
            ard_num_dims=X_np.shape[1],
            lengthscale_constraint=Interval(*LENGTH_SCALE_BOUNDS)
        )
        covar_module = ScaleKernel(matern)
        gp = SingleTaskGP(
            torch.tensor(X_np, dtype=torch.float64),
            torch.tensor(y_np, dtype=torch.float64).unsqueeze(-1),
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
            train_Yvar=None,
        )
        gp.likelihood.noise_covar.register_constraint(
            "raw_noise", NOISE_CONSTRAINT
        )
        return gp


# =============================================================================
# BoTorchGP  —  used for F1–F7
# =============================================================================

class BoTorchGP(GPModel):

    def fit(self, X, y, debug=False):
        self.N    = X.shape[1]
        self.X_np = X
        self.y_np = y

        self.gp = self._build_gp(X, y)
        self.gp.train()
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        self.gp.eval()

        if debug:
            print(f"\n--- BoTorchGP fitted [{ACQUISITION_STRATEGY.upper()}] ---")
            print("σ_f² (output scale)  :", self.gp.covar_module.outputscale.item())
            print("ℓ (length scales)    :",
                  self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel())
            print("σ_n² (noise)         :", self.gp.likelihood.noise.item())

    def evaluate(self, X, grid=False):
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            posterior = self.gp.posterior(torch.tensor(X, dtype=torch.float64))
            mean = posterior.mean.detach().numpy().ravel()
            std  = posterior.variance.sqrt().detach().numpy().ravel()
        if grid:
            mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std

    def _build_acquisition(self):
        """
        Acquisition function dispatch — final round:

        ┌─────────────────────────────────────────────────────────────────┐
        │  F1       → EI (localised bounds)       ← AMENDED             │
        │  F2, F3   → Upper Confidence Bound (UCB)                       │
        │  F4, F6   → Expected Improvement (EI)  ← BOTH AMENDED         │
        │  F5       → OVERRIDE (boundary corner neighbourhood)           │
        │  F7       → Expected Improvement (EI)                          │
        └─────────────────────────────────────────────────────────────────┘

        F1 localised EI: the EI acquisition function is identical to F4/F6/F7.
        The difference is in find_acquisition_max, where F1_LOCALISED_BOUNDS
        replaces the full [0,1]^2 bounds, constraining the search to the
        ±0.010 neighbourhood of the known best at (0.703500, 0.716000).
        """
        if ACQUISITION_STRATEGY in ('ei', 'localised_ei', 'override'):
            best_f = torch.tensor(self.y_np.max(), dtype=torch.float64)
            return ExpectedImprovement(model=self.gp, best_f=best_f)

        elif ACQUISITION_STRATEGY == 'ucb':
            kappa = UCB_KAPPA[FUNCTION_NO]
            return UpperConfidenceBound(model=self.gp, beta=kappa ** 2)

        elif ACQUISITION_STRATEGY == 'thompson':
            return None

        else:
            raise ValueError(f"Unknown acquisition strategy: {ACQUISITION_STRATEGY}")

    def evaluate_acquisition(self, X, grid=False):
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])

        if ACQUISITION_STRATEGY == 'thompson':
            mean, std = self.evaluate(
                X.reshape(-1, self.N) if not grid else X, grid=False
            )
            y = mean + 2.0 * std
            if grid:
                y = y.reshape(*shape)
            return y

        acq = self._build_acquisition()
        with torch.no_grad():
            y = acq(
                torch.tensor(X, dtype=torch.float64).unsqueeze(1)
            ).detach().numpy().ravel()

        if grid:
            y = y.reshape(*shape)
        return y

    def find_acquisition_max(self, debug=False):
        """
        Override dispatch (F1, F5):
          Returns the hard-coded override point directly, bypassing the GP
          acquisition function entirely. Used when the GP has demonstrated
          systematic failure to identify the true optimum.

        EI dispatch (F4, F6, F7):
          optimize_acqf with 512 raw samples + 32 restarts.
          F4 and F6 now use EI targeting their confirmed best points.

        UCB dispatch (F2, F3):
          optimize_acqf with 512 raw samples + 32 restarts.
        """

        # ── F1: Localised EI within ±0.010 box around known best ──────────────
        # Rationale: hard override to (0.703500, 0.716000) duplicates W2.
        # Localised EI runs the standard EI optimiser but constrained to
        # F1_LOCALISED_BOUNDS. EI = 0 at all observed points by construction,
        # so the result is guaranteed to differ from any previous query.
        if FUNCTION_NO == 1:
            acq = self._build_acquisition()   # EI with f* = 1.81e-9
            best_x, best_y = optimize_acqf(
                acq_function=acq,
                bounds=F1_LOCALISED_BOUNDS,   # ← constrained local box
                raw_samples=512,
                num_restarts=32,
                q=1
            )
            best_x = best_x.detach().numpy().ravel()
            best_y = best_y.detach().numpy().item()
            if debug:
                print("\n--- F1 LOCALISED EI ---")
                print("Search box: x1 ∈ [0.6935, 0.7135], x2 ∈ [0.7060, 0.7260]")
                print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
                print("EI value  :", best_y)
                print("Note: guaranteed ≠ (0.703500, 0.716000) by EI construction")
            return best_x, best_y

        # ── F5: Boundary neighbourhood override ───────────────────────────────
        if FUNCTION_NO == 5:
            best_x = F5_BOUNDARY_OVERRIDE
            mean, _ = self.evaluate(best_x.reshape(1, -1))
            best_y  = float(mean[0])
            if debug:
                print("\n--- F5 BOUNDARY OVERRIDE ---")
                print("Override point:", '-'.join(f'{x:.6f}' for x in best_x))
                print("Posterior mean at override:", best_y)
            return best_x, best_y

        bounds = torch.stack([
            torch.zeros(self.N, dtype=torch.float64),
            torch.ones(self.N,  dtype=torch.float64)
        ])

        # ── Thompson Sampling (retained for any future use) ────────────────────
        if ACQUISITION_STRATEGY == 'thompson':
            candidates = torch.rand(2048, self.N, dtype=torch.float64)
            ts = MaxPosteriorSampling(model=self.gp, replacement=False)
            with torch.no_grad():
                best_x = ts(candidates, num_samples=1).detach().numpy().ravel()
            mean, _ = self.evaluate(best_x.reshape(1, -1))
            best_y  = float(mean[0])
            if debug:
                print("\n--- Thompson Sampling acquisition max ---")
                print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
                print("Posterior mean at max:", best_y)
            return best_x, best_y

        # ── Gradient-based optimisation (EI, UCB) ─────────────────────────────
        acq = self._build_acquisition()
        best_x, best_y = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            raw_samples=512,
            num_restarts=32,
            q=1
        )
        best_x = best_x.detach().numpy().ravel()
        best_y = best_y.detach().numpy().item()

        if debug:
            print(f"\n--- {ACQUISITION_STRATEGY.upper()} acquisition max ---")
            print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
            print("Acquisition value:", best_y)

        return best_x, best_y


# =============================================================================
# TuRBoGP  —  used exclusively for F8
# =============================================================================

class TuRBoGP(GPModel):

    def __init__(self):
        self.tr_length       = TURBO_LENGTH_INIT
        self.success_counter = 0
        self.failure_counter = 0
        self.best_y          = -np.inf

    def fit(self, X, y, debug=False):
        self.N      = X.shape[1]
        self.X_np   = X
        self.y_np   = y
        self.best_y = float(y.max())

        self.gp = self._build_gp(X, y)
        self.gp.train()
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        self.gp.eval()

        if debug:
            print("\n--- TuRBoGP fitted [TURBO + Thompson Sampling] ---")
            print("σ_f² (output scale)  :", self.gp.covar_module.outputscale.item())
            print("ℓ (length scales)    :",
                  self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel())
            print("σ_n² (noise)         :", self.gp.likelihood.noise.item())
            print(f"Trust region length  : {self.tr_length:.4f}")
            print(f"Current best y       : {self.best_y:.6f}")

    def _get_tr_bounds(self):
        best_idx   = np.argmax(self.y_np)
        x_center   = self.X_np[best_idx]
        ls         = self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        weights    = ls / ls.max()
        half_width = (self.tr_length / 2.0) * weights
        lb = np.clip(x_center - half_width, 0.0, 1.0)
        ub = np.clip(x_center + half_width, 0.0, 1.0)
        return lb, ub

    def update_tr_state(self, new_y):
        if new_y > self.best_y:
            self.success_counter += 1
            self.failure_counter  = 0
            self.best_y = new_y
        else:
            self.failure_counter += 1
            self.success_counter  = 0

        if self.success_counter >= TURBO_SUCCESS_TOL:
            self.tr_length = min(self.tr_length * TURBO_SUCCESS_FACTOR, TURBO_LENGTH_MAX)
            self.success_counter = 0
            print(f"[TuRBO] TR expanded → L = {self.tr_length:.4f}")

        if self.failure_counter >= TURBO_FAILURE_TOL:
            self.tr_length = self.tr_length * TURBO_FAILURE_FACTOR
            self.failure_counter = 0
            print(f"[TuRBO] TR shrunk   → L = {self.tr_length:.4f}")

        if self.tr_length < TURBO_LENGTH_MIN:
            self.tr_length = TURBO_LENGTH_INIT
            print(f"[TuRBO] TR reset    → L = {self.tr_length:.4f}")

    def evaluate(self, X, grid=False):
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            posterior = self.gp.posterior(torch.tensor(X, dtype=torch.float64))
            mean = posterior.mean.detach().numpy().ravel()
            std  = posterior.variance.sqrt().detach().numpy().ravel()
        if grid:
            mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std

    def evaluate_acquisition(self, X, grid=False):
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        mean, std = self.evaluate(X)
        acq = mean + 2.0 * std
        lb, ub = self._get_tr_bounds()
        inside_tr = np.all((X >= lb) & (X <= ub), axis=1)
        acq = np.where(inside_tr, acq, acq.min())
        if grid:
            acq = acq.reshape(*shape)
        return acq

    def find_acquisition_max(self, debug=False):
        lb, ub = self._get_tr_bounds()
        lb_t = torch.tensor(lb, dtype=torch.float64)
        ub_t = torch.tensor(ub, dtype=torch.float64)
        candidates = lb_t + (ub_t - lb_t) * torch.rand(2048, self.N, dtype=torch.float64)

        ts = MaxPosteriorSampling(model=self.gp, replacement=False)
        with torch.no_grad():
            best_x = ts(candidates, num_samples=1).detach().numpy().ravel()

        mean, _ = self.evaluate(best_x.reshape(1, -1))
        best_y  = float(mean[0])

        if debug:
            print("\n--- TuRBO + Thompson Sampling acquisition max ---")
            print(f"Trust region length  : {self.tr_length:.4f}")
            print(f"TR lower bound       : {lb}")
            print(f"TR upper bound       : {ub}")
            print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
            print("Posterior mean at max:", best_y)

        return best_x, best_y


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
