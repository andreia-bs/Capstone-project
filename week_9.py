
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
from botorch.generation import MaxPosteriorSampling  # Thompson Sampling — F4, F6

WEEK_NO = 11

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
#   ARD (Automatic Relevance Determination) is enabled for all functions with
#   N>1 so that irrelevant input dimensions are automatically down-weighted.
# =============================================================================

LENGTH_SCALE_BOUNDS = {
    1: (0.005, 0.05),  # AMENDED: Much tighter — force GP to model sharp peak
                       # Previous (0.02, 0.3) was too wide; MES repeatedly
                       # misfired by fitting a smooth surface over a near-
                       # singular peak. Tight bounds force short length scales,
                       # making the GP sensitive to the narrow peak region.
    2: (0.05, 0.5),    # Moderate 1D
    3: (0.1,  1.0),    # Smooth, wider
    4: (0.05, 0.5),    # Jagged 2D
    5: (0.1,  1.5),    # Smooth, very wide
    6: (0.05, 0.75),   # Moderate multi-dim
    7: (0.05, 1.0),    # HP tuning output — ARD important  [Ref 4]
    8: (0.05, 1.0),    # 8D — TuRBO manages local scales  [Ref 2]
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
#   'ucb'            → Upper Confidence Bound — F2, F3
#   'thompson'       → Thompson Sampling — F4, F6
#   'ei'             → Expected Improvement — F1 (AMENDED), F5, F7
#   'turbo'          → Trust-Region BO — F8                        [Ref 2]
#
# AMENDMENTS vs previous version:
#   F1: 'entropy_search' → 'ei'
#       Rationale: MES has consistently misfired across multiple modules,
#       producing outputs as low as 8.80e-83 despite a known best of
#       1.81e-9 at (0.703500, 0.716000). MES's information-theoretic
#       exploration bonus is too aggressive for a function this sharp —
#       it keeps directing queries to uncertain regions far from the peak.
#       EI with the tightened length-scale bounds will exploit the known
#       best region directly, computing expected improvement over the
#       current best f* = 1.81e-9.
#
#   F2: strategy unchanged ('ucb'), but κ reduced from 3.0 → 1.0
#       Rationale: UCB with κ=3.0 sent W9 query to (0.957, 0.997),
#       returning only 0.057 against an all-time best of 0.628. The
#       exploration bonus dominated, pulling the query to an untested
#       corner. Reducing κ to 1.0 forces exploitation near the confirmed
#       best at (0.716, 0.403).
#
#   F5: strategy unchanged ('ei'), but a manual override is applied
#       in find_acquisition_max to avoid re-proposing (1,1,1,1).
#       Rationale: EI correctly identifies (1,1,1,1) as the best known
#       point but cannot propose beyond the boundary. The override
#       queries (0.990, 0.990, 0.990, 0.990) to test whether the
#       optimum extends slightly inside the boundary corner.
# =============================================================================

ACQUISITION_STRATEGY = {
    1: 'ei',          # AMENDED: EI replaces MES — exploit known peak region
    2: 'ucb',         # Unchanged — but κ reduced (see UCB_KAPPA)
    3: 'ucb',         # UCB with smooth kernel
    4: 'thompson',    # Thompson Sampling avoids UCB over-exploration
    5: 'ei',          # EI exploits confirmed best (1,1,1,1) region
    6: 'thompson',    # TS introduces diversity; avoids EI repeat
    7: 'ei',          # EI is the AutoML/HPO community standard  [Ref 4]
    8: 'turbo',       # TuRBO for 8D — global GP unreliable      [Ref 2]
}[FUNCTION_NO]

# UCB κ — only used when strategy is 'ucb'
UCB_KAPPA = {
    2: 1.0,   # AMENDED: reduced from 3.0 — force exploitation near (0.716, 0.403)
    3: 2.0,   # Slightly more exploitative (smoother function)
}

# =============================================================================
# ── F5 BOUNDARY OVERRIDE ──────────────────────────────────────────────────────
# =============================================================================
# Rationale:
#   F5's confirmed optimum is at (1,1,1,1) = 8662.48. EI cannot propose
#   beyond the [0,1]^4 boundary, so it will re-propose (1,1,1,1) — a
#   duplicate of W8. Instead, we override the acquisition max to query
#   (0.99, 0.99, 0.99, 0.99), testing whether the function continues to
#   increase right up to the boundary or peaks slightly inside it.
#   If this returns a value close to 8662.48, the corner is confirmed as
#   the true optimum. If it returns significantly less, the optimum is
#   exactly at the boundary corner.
# =============================================================================

F5_BOUNDARY_OVERRIDE = np.array([0.99, 0.99, 0.99, 0.99])

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

TURBO_LENGTH_INIT    = 0.4   # Initial TR side length (fraction of unit hypercube)
TURBO_LENGTH_MIN     = 0.01  # Minimum TR length before restart
TURBO_LENGTH_MAX     = 1.6   # Maximum TR length
TURBO_SUCCESS_TOL    = 3     # Consecutive successes before expanding TR
TURBO_FAILURE_TOL    = 4     # Consecutive failures before shrinking TR
TURBO_SUCCESS_FACTOR = 2.0   # TR expansion multiplier
TURBO_FAILURE_FACTOR = 0.5   # TR contraction multiplier

# =============================================================================
# ── NOISE FLOOR CONFIGURATION ─────────────────────────────────────────────────
# =============================================================================

NOISE_CONSTRAINT = {
    1: Interval(1e-6, 1e-3),   # AMENDED: tighter — prevent noise absorption
                                # on sharp F1. Previous (1e-5, 1e-2) still
                                # allowed too much noise, flattening the
                                # posterior near the peak.
    2: Interval(1e-5, 1e-1),   # Moderate
    3: Interval(1e-5, 1e-1),   # Moderate
    4: Interval(1e-5, 1e-2),   # Tight — jagged F4
    5: Interval(1e-5, 1e-1),   # Moderate — EI used; relaxed noise ok
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

    if ACQUISITION_STRATEGY == 'turbo':
        model = TuRBoGP()
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
                   color='green', label='Acquisition max')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'GP posterior — Function {FUNCTION_NO} [{ACQUISITION_STRATEGY.upper()}]')
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
                linestyle='--', label='Acquisition max')
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
        """
        Constructs a SingleTaskGP with:
          - Matérn kernel (ν from NU_SMOOTHNESS, ARD enabled)
          - ScaleKernel wrapper (learns output variance σ_f²)
          - Standardize outcome transform (zero-mean normalisation)
          - Per-function noise constraint (NOISE_CONSTRAINT)
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
        Acquisition function dispatch:

        ┌─────────────────────────────────────────────────────────────────┐
        │  F1       → Expected Improvement (EI)       ← AMENDED          │
        │  F2, F3   → Upper Confidence Bound (UCB)                       │
        │  F4, F6   → Thompson Sampling (TS)                             │
        │  F5, F7   → Expected Improvement (EI)                          │
        └─────────────────────────────────────────────────────────────────┘

        F1 amendment: EI replaces MES. MES repeatedly misfired (W9 output:
        8.80e-83). EI with tightened length-scale bounds will exploit the
        known best region at (0.703500, 0.716000) where f* = 1.81e-9.

        F2 amendment: UCB κ reduced from 3.0 to 1.0. W9's κ=3.0 sent the
        query to (0.957, 0.997) returning only 0.057. Lower κ forces
        exploitation near the confirmed best at (0.716, 0.403).
        """

        # ── F1, F5, F7: EXPECTED IMPROVEMENT ──────────────────────────────────
        # F1 AMENDMENT: EI replaces MES for direct exploitation of known peak.
        if ACQUISITION_STRATEGY == 'ei':
            best_f = torch.tensor(self.y_np.max(), dtype=torch.float64)
            return ExpectedImprovement(model=self.gp, best_f=best_f)

        # ── F2, F3: UPPER CONFIDENCE BOUND ────────────────────────────────────
        elif ACQUISITION_STRATEGY == 'ucb':
            kappa = UCB_KAPPA[FUNCTION_NO]
            return UpperConfidenceBound(model=self.gp, beta=kappa ** 2)

        # ── F4, F6: THOMPSON SAMPLING ──────────────────────────────────────────
        elif ACQUISITION_STRATEGY == 'thompson':
            return None  # Handled separately in find_acquisition_max

        else:
            raise ValueError(f"Unknown acquisition strategy: {ACQUISITION_STRATEGY}")

    def evaluate_acquisition(self, X, grid=False):
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])

        # Thompson Sampling (F4, F6): no closed-form; use UCB(κ=2) as proxy
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
        F4, F6 (Thompson Sampling): MaxPosteriorSampling over 2048 candidates.
        F1, F5, F7 (EI): optimize_acqf, 512 raw + 32 restarts.
        F2, F3 (UCB): optimize_acqf, 512 raw + 32 restarts.

        F5 BOUNDARY OVERRIDE:
          EI will re-propose (1,1,1,1) since it is the confirmed best and
          lies at the boundary — EI cannot propose beyond [0,1]^4.
          Instead, we return F5_BOUNDARY_OVERRIDE = (0.99, 0.99, 0.99, 0.99)
          to test whether the optimum extends slightly inside the boundary.
        """
        # ── F5: Boundary override ──────────────────────────────────────────────
        if FUNCTION_NO == 5:
            best_x = F5_BOUNDARY_OVERRIDE
            mean, _ = self.evaluate(best_x.reshape(1, -1))
            best_y  = float(mean[0])
            if debug:
                print("\n--- F5 boundary override ---")
                print("Override point:", '-'.join(f'{x:.6f}' for x in best_x))
                print("Posterior mean at override:", best_y)
            return best_x, best_y

        bounds = torch.stack([
            torch.zeros(self.N, dtype=torch.float64),
            torch.ones(self.N,  dtype=torch.float64)
        ])

        # ── Thompson Sampling (F4, F6) ─────────────────────────────────────────
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
# Reference: Eriksson et al. (2019) — TuRBO [Ref 2]
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
