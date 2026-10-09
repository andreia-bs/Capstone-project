# =============================================================================
# BESPOKE BAYESIAN OPTIMISATION — CAPSTONE PROJECT
# =============================================================================
# This script replaces the single-strategy BO pipeline with function-specific
# acquisition functions, kernel configurations, and optimisation strategies.
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
from scipy.stats import norm as scipy_norm   # used for EI and ES

import warnings

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    UpperConfidenceBound,
    ExpectedImprovement,
    qMaxValueEntropy,          # Max-value Entropy Search — F1  [Ref 1]
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.constraints import Interval
from botorch.generation import MaxPosteriorSampling  # Thompson Sampling — F4

WEEK_NO = 7

# Select the function number from command line argument
FUNCTION_NO = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# =============================================================================
# ── PER-FUNCTION KERNEL CONFIGURATION ────────────────────────────────────────
# =============================================================================
# Rationale:
#   Length-scale bounds are kept tight for sharp/low-dim functions (F1, F2, F4)
#   and relaxed for smoother/higher-dim functions (F5, F7, F8).
#   ν (nu) controls Matérn smoothness:
#     ν=1.5 → once-differentiable (jagged)
#     ν=2.5 → twice-differentiable (smooth)
# Higher ν→ the modeled function is smoother and more regular. 
# Lower ν→ the function is rougher and more jagged.
#   ARD (Automatic Relevance Determination) is enabled for all functions with
#   N>1 so that irrelevant input dimensions are automatically down-weighted.
# =============================================================================

LENGTH_SCALE_BOUNDS = {
    1: (0.02, 0.3),   # Sharp 1D — keep tight
    2: (0.05, 0.5),   # Moderate 1D
    3: (0.1,  1.0),   # Smooth, wider
    4: (0.05, 0.5),   # Jagged 2D
    5: (0.1,  1.5),   # Smooth, very wide — aggressive exploration
    6: (0.05, 0.75),  # Moderate multi-dim
    7: (0.05, 1.0),   # HP tuning output — ARD important  [Ref 4]
    8: (0.05, 1.0),   # 8D — TuRBO manages local scales  [Ref 2]
}[FUNCTION_NO]



NU_SMOOTHNESS = {
    1: 1.5,   # Sharp/noisy → once-differentiable
    2: 1.5,   # Moderate sharpness
    3: 2.5,   # Smooth
    4: 1.5,   # Jagged
    5: 2.5,   # Smooth, large space
    6: 2.5,   # Smooth moderate-dim
    7: 2.5,   # HP tuning — smooth response surface assumed
    8: 2.5,   # 8D — smooth assumed; TuRBO handles locality  [Ref 2]
}[FUNCTION_NO]

# =============================================================================
# ── PER-FUNCTION ACQUISITION STRATEGY ────────────────────────────────────────
# =============================================================================
# Choices:
#   'entropy_search' → Max-Value Entropy Search (MES) — F1         [Ref 1]
#   'ucb'            → Upper Confidence Bound — F2, F3, F5
#   'thompson'       → Thompson Sampling — F4
#   'ei'             → Expected Improvement — F6, F7
#   'turbo'          → Trust-Region BO — F8                        [Ref 2]
# =============================================================================

ACQUISITION_STRATEGY = {
    1: 'entropy_search',  # UCB misfires on F1; ES is information-theoretic  [Ref 1]
    2: 'ucb',             # Standard UCB works well for moderate 1D
    3: 'ucb',             # UCB with smooth kernel
    4: 'thompson',        # Thompson Sampling avoids UCB over-exploration
    5: 'ucb',             # High κ forces exploration of vast space
    6: 'ei',              # EI balances exploitation robustly for moderate-dim
    7: 'ei',              # EI is the AutoML/HPO community standard  [Ref 4]
    8: 'turbo',           # TuRBO for 8D — global GP unreliable      [Ref 2]
}[FUNCTION_NO]

# UCB κ — only used when strategy is 'ucb'
UCB_KAPPA = {
    2: 3.0,     # Balanced
    3: 2.0,     # Slightly more exploitative (smoother function)
    5: 1000.0,  # Aggressive exploration — vast unexplored space
}

# =============================================================================
# ── TURBO HYPERPARAMETERS (F8 only) ──────────────────────────────────────────
# =============================================================================
# Rationale [Ref 2]:
#   TuRBO maintains a trust region (TR) of side length L around the current
#   best point. Candidates are drawn only within the TR, preventing the GP
#   from being queried in regions where it has no data (a key failure mode
#   of standard BO in 8D).
#   L shrinks on failure and grows on success, providing adaptive locality.
# =============================================================================

TURBO_LENGTH_INIT   = 0.4   # Initial TR side length (fraction of unit hypercube)
TURBO_LENGTH_MIN    = 0.01  # Minimum TR length before restart
TURBO_LENGTH_MAX    = 1.6   # Maximum TR length
TURBO_SUCCESS_TOL   = 3     # Consecutive successes before expanding TR
TURBO_FAILURE_TOL   = 4     # Consecutive failures before shrinking TR
TURBO_SUCCESS_FACTOR = 2.0  # TR expansion multiplier
TURBO_FAILURE_FACTOR = 0.5  # TR contraction multiplier


# =============================================================================
# ── NOISE FLOOR CONFIGURATION ─────────────────────────────────────────────────
# =============================================================================
# Rationale:
#   F1 and F4 appear noisy/jagged. Adding a tighter noise constraint prevents
#   the GP from absorbing all variance as observation noise, which would
#   flatten the posterior and suppress acquisition peaks.
#   F5 uses a relaxed noise floor because the function is smooth but the
#   evaluations may carry numerical noise from the large search space.
# =============================================================================

NOISE_CONSTRAINT = {
    1: Interval(1e-5, 1e-2),   # Tight — prevent noise absorption on sharp F1
    2: Interval(1e-5, 1e-1),   # Moderate
    3: Interval(1e-5, 1e-1),   # Moderate
    4: Interval(1e-5, 1e-2),   # Tight — jagged F4
    5: Interval(1e-5, 1e-1),   # Relaxed
    6: Interval(1e-5, 1e-1),   # Moderate
    7: Interval(1e-5, 1e-1),   # Moderate — HP tuning outputs can be noisy
    8: Interval(1e-5, 1e-1),   # Moderate
}[FUNCTION_NO]


# =============================================================================
# MAIN
# =============================================================================

def main():
    X, y = read_data(debug=False, plot_datapoints=False)
    N = X.shape[1]

    # Dispatch to the correct model based on the per-function strategy
    if ACQUISITION_STRATEGY == 'turbo':
        # ── F8: TuRBO ─────────────────────────────────────────────────────────
        # TuRBO maintains its own state (TR length, success/failure counters)
        # and wraps a standard GP internally.
        model = TuRBoGP()
    else:
        # ── F1–F7: Standard BoTorch GP with bespoke acquisition ───────────────
        model = BoTorchGP()

    model.fit(X, y, debug=True)
    model.find_acquisition_max(debug=True)
    plot_gp_results(model, X, y)


# =============================================================================
# DATA LOADING  (unchanged from original)
# =============================================================================

def read_data(debug=False, plot_datapoints=True):
    # --- Load the initial data ---
    X = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_inputs.npy")
    y = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_outputs.npy")
    N = X.shape[1]

    # --- Add data from previous weeks ---
    for week in range(1, WEEK_NO):
        with open(f"./week{week}_inputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_values = np.array([float(x) for x in line.strip().split('-')])
            X = np.vstack((X, new_values))
        with open(f"./week{week}_outputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_value = float(line.strip())
            y = np.append(y, np.array(new_value))

    # --- Print the inputs and outputs for the selected function ---
    if debug:
        print("\n--- Data points ---")
        for Xi, yi in zip(X, y):
            print(Xi, yi)

    # --- Plot the outputs with colors ---
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
# PLOTTING  (unchanged from original, supports 1D–8D)
# =============================================================================

# --- Plot the GP and acquisition function results ---
def plot_gp_results(model, X, y):
    N = X.shape[1]
    # --- Plotting for 2D cases ---
    if N == 1:
        linspace   = np.linspace(0, 1, 100)
        mean, std  = model.evaluate(linspace.reshape(-1, 1), grid=False)
        acquisition = model.evaluate_acquisition(linspace.reshape(-1, 1), grid=False)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig, ax = plt.subplots(figsize=(10, 8))
        # Scatter of the actual data points
        ax.scatter(X[:-WEEK_NO + 1], y[:-WEEK_NO + 1], color='black', marker='o', s=40)
        ax.scatter(X[-WEEK_NO + 1:], y[-WEEK_NO + 1:], color='black', marker='s', s=40)

        # Posterior mean
        ax.plot(linspace, mean, linewidth=2, label='Posterior mean')

        # Upper and lower std bounds surfaces
        ax.fill_between(linspace, mean - std, mean + std, alpha=0.2, label='Posterior std')

        # Acquisition function
        ax.plot(linspace, acquisition, color='green', linewidth=2, label='Acquisition')

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X, acquisition_max_y, marker='*', s=200,
                   color='green', label='Acquisition max')

        # Set labels and title
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'GP posterior — Function {FUNCTION_NO} '
                     f'[{ACQUISITION_STRATEGY.upper()}]')
        ax.legend(); plt.tight_layout(); plt.show()

    # --- Plotting for 3D cases ---
    elif N == 2:
        linspace = np.linspace(0, 1, 25)
        mesh     = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        
        # Scatter of the  data points
        ax.scatter(X[:-WEEK_NO + 1, 0], X[:-WEEK_NO + 1, 1], y[:-WEEK_NO + 1],
                   marker='o', color='black', s=50, depthshade=True)
        ax.scatter(X[-WEEK_NO + 1:, 0], X[-WEEK_NO + 1:, 1], y[-WEEK_NO + 1:],
                   marker='s', color='black', s=50, depthshade=True)

        # Posterior mean surface
        ax.plot_surface(mesh[0], mesh[1], mean, cmap='rainbow', alpha=0.8,
                        linewidth=0, antialiased=True)

        # Upper and lower std bounds surfaces
        ax.plot_surface(mesh[0], mesh[1], mean + std, color='lightgray',
                        alpha=0.2, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], mean - std, color='lightgray',
                        alpha=0.2, linewidth=0)

        # Acquisition function surface
        ax.plot_surface(mesh[0], mesh[1], acquisition, color='green',
                        alpha=0.2, linewidth=0)

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_y,
                   color='green', s=200, marker='*')

        # Set labels and title
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
        ax.set_title(f'GP posterior — Function {FUNCTION_NO} '
                     f'[{ACQUISITION_STRATEGY.upper()}]')
        plt.tight_layout(); plt.show()

    # --- Plotting for 4D cases ---
    elif N == 3:
        linspace = np.linspace(0, 1, 20)
        mesh     = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, _ = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        # Scatter of the actual data points with colour and size based on output value
        sizes = 40 * (y - y.min()) / (y.max() - y.min() + 1e-9) + 10
        ax.scatter(X[:-WEEK_NO + 1, 0], X[:-WEEK_NO + 1, 1], X[:-WEEK_NO + 1, 2],
                   s=sizes[:-WEEK_NO + 1], c=y[:-WEEK_NO + 1], cmap='rainbow',
                   vmin=y.min(), vmax=y.max(), marker='o', depthshade=True)
        ax.scatter(X[-WEEK_NO + 1:, 0], X[-WEEK_NO + 1:, 1], X[-WEEK_NO + 1:, 2],
                   s=sizes[-WEEK_NO + 1:], c=y[-WEEK_NO + 1:], cmap='rainbow',
                   vmin=y.min(), vmax=y.max(), marker='s', depthshade=True)

        # Plot the acquisition function as a scatter plot
        ax.scatter(mesh[0], mesh[1], mesh[2], c=acquisition, cmap='viridis',
                   marker='o', s=100, alpha=0.4, edgecolors='none')

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_X[2],
                   color='green', s=200, marker='*')

        # Set labels and title
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
        ax.set_title(f'Observed data — Function {FUNCTION_NO} '
                     f'[{ACQUISITION_STRATEGY.upper()}]')
        plt.tight_layout(); plt.show()

    # --- Plotting for higher-dimensional cases (5D+) ---
    else:  # N > 3 — parallel coordinates
        acquisition_max_X, _ = model.find_acquisition_max()
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize y for colormap
        norm = plt.Normalize(y.min(), y.max())
        cmap = plt.cm.rainbow

        x_axis = np.arange(N)

        # Plot each point as a line in the parallel coordinates plot
        for i in range(X.shape[0] - WEEK_NO + 1):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=1, alpha=0.6)
        for i in range(X.shape[0] - WEEK_NO + 1, X.shape[0]):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=2,
                    linestyle='--', alpha=0.6)

        # Plot the acquisition function maximum point as a dashed line in the parallel coordinates plot
        ax.plot(x_axis, acquisition_max_X, color='green', linewidth=3,
                linestyle='--', label='Acquisition max')

        # Set labels and title
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f'x[{d}]' for d in range(1, N + 1)])
        ax.set_title(f'Parallel coordinates — Function {FUNCTION_NO} '
                     f'[{ACQUISITION_STRATEGY.upper()}]')
        ax.set_ylabel('Input value'); ax.legend()

        # Colorbar to show what color maps to what y value
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='y')

        # Display the plot
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

    # ------------------------------------------------------------------
    # Shared helper: build a BoTorch SingleTaskGP with the per-function
    # kernel and noise configuration.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_gp(X_np, y_np):
        """
        Constructs a SingleTaskGP with:
          - Matérn kernel (ν from NU_SMOOTHNESS, ARD enabled)
          - ScaleKernel wrapper (learns output variance σ_f²)
          - Standardize outcome transform (zero-mean normalisation)
          - Per-function noise constraint (NOISE_CONSTRAINT)
        The Standardize transform is critical: without it, unexplored
        regions default to the raw output mean rather than 0, which
        biases UCB/EI toward already-explored areas.
        """
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
            # Per-function noise floor — prevents noise absorption
            train_Yvar=None,
        )
        # Override the default noise constraint with our per-function one
        gp.likelihood.noise_covar.register_constraint(
            "raw_noise", NOISE_CONSTRAINT
        )
        return gp


# =============================================================================
# BoTorchGP  —  used for F1–F7
# =============================================================================
# Each function selects its acquisition function via ACQUISITION_STRATEGY.
# The GP construction is shared; only the acquisition layer differs.
# =============================================================================

class BoTorchGP(GPModel):

    # ──────────────────────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, X, y, debug=False):
        """
        Fits the GP via marginal log-likelihood (MLL) maximisation using
        L-BFGS-B internally (BoTorch default). n_restarts is implicitly
        handled by fit_gpytorch_mll's multi-start optimiser.
        """
        self.N  = X.shape[1]
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

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE POSTERIOR
    # ──────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD ACQUISITION FUNCTION  (dispatched per function)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_acquisition(self):
        """
        Dispatches to the correct acquisition function based on
        ACQUISITION_STRATEGY for the current FUNCTION_NO.

        ┌─────────────────────────────────────────────────────────────────┐
        │  F1 → Max-Value Entropy Search (MES)                [Ref 1]    │
        │  F2, F3, F5 → Upper Confidence Bound (UCB)                     │
        │  F4 → Thompson Sampling                                         │
        │  F6, F7 → Expected Improvement (EI)                 [Ref 4]    │
        └─────────────────────────────────────────────────────────────────┘
        """

        # ── F1: MAX-VALUE ENTROPY SEARCH ──────────────────────────────────────
        # Rationale [Ref 1]:
        #   UCB on F1 has repeatedly over-explored because its exploration
        #   bonus (κ·σ) dominates in regions where the GP is uncertain but
        #   the true function has already been well-characterised.
        #   MES instead measures the expected reduction in entropy about the
        #   *maximum value* f* — it asks "how much will this query tell me
        #   about where the optimum is?" rather than blindly rewarding
        #   uncertainty. This makes it far more sample-efficient on sharp,
        #   noisy 1D functions.
        #   BoTorch's qMaxValueEntropy approximates MES via fantasy samples
        #   drawn from the posterior predictive over a candidate set.
        if ACQUISITION_STRATEGY == 'entropy_search':
            candidate_set = torch.rand(
                512, self.N, dtype=torch.float64
            )  # Monte-Carlo candidate pool for entropy estimation
            return qMaxValueEntropy(
                model=self.gp,
                candidate_set=candidate_set,
                num_fantasies=128,   # Number of fantasy samples — more = better estimate, slower
                use_gumbel=True      # Gumbel approximation for speed (recommended by BoTorch)
            )

        # ── F2, F3, F5: UPPER CONFIDENCE BOUND ───────────────────────────────
        # Rationale:
        #   UCB remains appropriate for these functions.
        #   F5 uses κ=1000 to force near-pure exploration of a vast,
        #   under-sampled space. F2 and F3 use moderate κ values.
        #   Note: BoTorch's UCB uses β = κ² internally.
        elif ACQUISITION_STRATEGY == 'ucb':
            kappa = UCB_KAPPA[FUNCTION_NO]
            return UpperConfidenceBound(model=self.gp, beta=kappa ** 2)

        # ── F4: THOMPSON SAMPLING ─────────────────────────────────────────────
        # Rationale:
        #   F4 is a jagged 2D function where UCB's deterministic exploration
        #   bonus causes it to revisit the same uncertain regions repeatedly.
        #   Thompson Sampling (TS) draws a random function from the posterior
        #   and optimises *that* — this introduces natural diversity in the
        #   query locations without a manually tuned κ.
        #   TS is asymptotically optimal and avoids the over-exploration
        #   failure mode of UCB on multi-modal landscapes.
        #   Implementation: we use MaxPosteriorSampling, which draws a
        #   sample path and returns its argmax over a discrete candidate set.
        elif ACQUISITION_STRATEGY == 'thompson':
            return None  # Thompson Sampling is handled separately in find_acquisition_max

        # ── F6, F7: EXPECTED IMPROVEMENT ─────────────────────────────────────
        # Rationale:
        #   EI is the gold standard for moderate-dimensional problems where
        #   the function is smooth and the goal is to find the optimum
        #   efficiently. It computes the expected amount by which the next
        #   query will exceed the current best, naturally balancing
        #   exploration and exploitation without a tunable κ.
        #   For F7 (hyperparameter tuning), EI is the standard acquisition
        #   used in AutoML systems (e.g. SMAC, Spearmint) and is well-
        #   calibrated against OpenML benchmark performance ranges [Ref 4].
        elif ACQUISITION_STRATEGY == 'ei':
            best_f = torch.tensor(self.y_np.max(), dtype=torch.float64)
            return ExpectedImprovement(model=self.gp, best_f=best_f)

        else:
            raise ValueError(f"Unknown acquisition strategy: {ACQUISITION_STRATEGY}")

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE ACQUISITION
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_acquisition(self, X, grid=False):
        """
        Evaluates the acquisition function on a grid or flat array of points.
        Thompson Sampling (F4) does not support pointwise evaluation, so we
        fall back to the posterior mean + std as a proxy for visualisation.
        """
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])

        # F4: TS has no closed-form pointwise value; use UCB(κ=2) as proxy
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

    # ──────────────────────────────────────────────────────────────────────────
    # FIND ACQUISITION MAXIMUM
    # ──────────────────────────────────────────────────────────────────────────
    def find_acquisition_max(self, debug=False):
        """
        Finds the point that maximises the acquisition function.

        ┌──────────────────────────────────────────────────────────────────┐
        │  F1 (MES), F2/3/5 (UCB), F6/7 (EI):                            │
        │    → optimize_acqf with 512 raw samples + 32 restarts           │
        │  F4 (Thompson Sampling):                                         │
        │    → MaxPosteriorSampling over 2048 discrete candidates         │
        └──────────────────────────────────────────────────────────────────┘

        For Thompson Sampling, we draw a single sample path from the GP
        posterior and return the candidate with the highest sampled value.
        This is equivalent to optimising a random draw from the posterior,
        which is the definition of TS.
        """
        bounds = torch.stack([
            torch.zeros(self.N,  dtype=torch.float64),
            torch.ones(self.N,   dtype=torch.float64)
        ])

        # ── F4: Thompson Sampling via MaxPosteriorSampling ────────────────────
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

        # ── All other strategies: gradient-based optimisation ─────────────────
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
# Reference: Eriksson et al. (2019) — TuRBO [Ref 2]
#
# Rationale:
#   F8 has 8 input dimensions. Standard global BO fails in high dimensions
#   because:
#     (a) The GP's covariance matrix becomes ill-conditioned with many
#         length scales to learn simultaneously.
#     (b) The acquisition function has exponentially many local maxima,
#         making gradient-based optimisation unreliable.
#     (c) The GP posterior is overconfident in unexplored regions, causing
#         the acquisition to waste queries far from the current best.
#
#   TuRBO addresses all three by restricting queries to a *trust region*
#   (TR) — a hyper-rectangle centred on the current best point, with side
#   length L. The TR:
#     • Shrinks when queries fail to improve (TURBO_FAILURE_TOL consecutive
#       failures → L ← L × TURBO_FAILURE_FACTOR).
#     • Grows when queries succeed (TURBO_SUCCESS_TOL consecutive successes
#       → L ← L × TURBO_SUCCESS_FACTOR).
#     • Resets to TURBO_LENGTH_INIT if L < TURBO_LENGTH_MIN.
#
#   Within the TR, we use Thompson Sampling (MaxPosteriorSampling) to
#   propose the next query — this is the original TuRBO formulation.
#   The GP is refitted from scratch each iteration using all data.
# =============================================================================

class TuRBoGP(GPModel):

    def __init__(self):
        # Trust-region state
        self.tr_length       = TURBO_LENGTH_INIT
        self.success_counter = 0
        self.failure_counter = 0
        self.best_y          = -np.inf

    # ──────────────────────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, X, y, debug=False):
        """
        Fits the GP on all available data (global fit).
        The trust region restricts *where* we query, not how we fit the GP.
        Using all data for fitting ensures the GP has the best possible
        posterior estimate within the TR.
        """
        self.N    = X.shape[1]
        self.X_np = X
        self.y_np = y
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

    # ──────────────────────────────────────────────────────────────────────────
    # TRUST REGION BOUNDS
    # ──────────────────────────────────────────────────────────────────────────
    def _get_tr_bounds(self):
        """
        Computes the trust region as a hyper-rectangle of side length
        self.tr_length centred on the current best point (x*).

        The TR is clipped to [0, 1]^N to stay within the valid input domain.
        The length scale of each dimension is taken into account:
          TR half-width in dim d = (tr_length / 2) × (ℓ_d / max(ℓ))
        This scales the TR anisotropically, giving more room in dimensions
        where the GP has learned a longer length scale (i.e. smoother dims).
        """
        best_idx = np.argmax(self.y_np)
        x_center = self.X_np[best_idx]

        # Anisotropic scaling using fitted length scales
        ls = self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        weights = ls / ls.max()  # normalise so the longest dim gets full width

        half_width = (self.tr_length / 2.0) * weights

        lb = np.clip(x_center - half_width, 0.0, 1.0)
        ub = np.clip(x_center + half_width, 0.0, 1.0)

        return lb, ub

    # ──────────────────────────────────────────────────────────────────────────
    # TRUST REGION STATE UPDATE
    # ──────────────────────────────────────────────────────────────────────────
    def update_tr_state(self, new_y):
        """
        Call this after each new observation to update the TR length.
        This method is intended to be called by the outer optimisation loop
        (not during the single-query planning phase).

        Args:
            new_y (float): The observed output of the latest query.
        """
        if new_y > self.best_y:
            self.success_counter += 1
            self.failure_counter  = 0
            self.best_y = new_y
        else:
            self.failure_counter += 1
            self.success_counter  = 0

        if self.success_counter >= TURBO_SUCCESS_TOL:
            self.tr_length = min(self.tr_length * TURBO_SUCCESS_FACTOR,
                                 TURBO_LENGTH_MAX)
            self.success_counter = 0
            print(f"[TuRBO] TR expanded → L = {self.tr_length:.4f}")

        if self.failure_counter >= TURBO_FAILURE_TOL:
            self.tr_length = self.tr_length * TURBO_FAILURE_FACTOR
            self.failure_counter = 0
            print(f"[TuRBO] TR shrunk   → L = {self.tr_length:.4f}")

        if self.tr_length < TURBO_LENGTH_MIN:
            self.tr_length = TURBO_LENGTH_INIT
            print(f"[TuRBO] TR reset    → L = {self.tr_length:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE POSTERIOR
    # ──────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE ACQUISITION  (visualisation only — UCB proxy within TR)
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_acquisition(self, X, grid=False):
        """
        TuRBO uses Thompson Sampling internally, which has no closed-form
        pointwise acquisition value. For plotting purposes we use a UCB
        proxy (κ=2) masked to zero outside the trust region.
        """
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])

        mean, std = self.evaluate(X)
        acq = mean + 2.0 * std

        # Zero out points outside the trust region (for visualisation clarity)
        lb, ub = self._get_tr_bounds()
        inside_tr = np.all((X >= lb) & (X <= ub), axis=1)
        acq = np.where(inside_tr, acq, acq.min())

        if grid:
            acq = acq.reshape(*shape)
        return acq

    # ──────────────────────────────────────────────────────────────────────────
    # FIND ACQUISITION MAXIMUM  (Thompson Sampling within TR)
    # ──────────────────────────────────────────────────────────────────────────
    def find_acquisition_max(self, debug=False):
        """
        Proposes the next query point using Thompson Sampling restricted to
        the current trust region.

        Steps:
          1. Compute TR bounds (anisotropic, centred on current best).
          2. Draw 2048 quasi-random candidates uniformly within the TR.
          3. Use MaxPosteriorSampling to draw one sample path and return
             the candidate with the highest sampled value.

        This is the core of TuRBO [Ref 2]: by restricting candidates to the
        TR, we ensure the GP is queried only where it has sufficient data to
        be reliable, avoiding the pathological over-exploration that standard
        BO exhibits in 8D.
        """
        lb, ub = self._get_tr_bounds()

        # Sobol-like uniform sampling within the TR
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