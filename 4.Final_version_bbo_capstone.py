# =============================================================================
# BESPOKE BAYESIAN OPTIMISATION — CAPSTONE PROJECT
# FINAL ROUND VERSION — MODULE 24
# =============================================================================
#
# PURPOSE:
#   This script implements a bespoke Bayesian Optimisation (BO) pipeline
#   for the BBO capstone project. It optimises eight unknown black-box
#   functions by fitting a Gaussian Process (GP) surrogate model to all
#   accumulated data and using an acquisition function to propose the next
#   query point.
#
# HOW TO USE:
#   1. Set FUNCTION_NO (1–8) and WEEK_NO in the CONFIGURATION section below
#   2. Ensure your data files are in the correct directory structure:
#        ./initial_data/function_{N}/initial_inputs.npy
#        ./initial_data/function_{N}/initial_outputs.npy
#        ./week{N}_inputs.txt   (one line per function, hyphen-separated)
#        ./week{N}_outputs.txt  (one line per function)
#   3. Run: python bbo_capstone_final_round.py [function_number]
#      Or:  python bbo_capstone_final_round.py   (uses FUNCTION_NO below)
#
# ACQUISITION STRATEGIES (per function):
#   F1 → Localised EI  : EI constrained to ±0.010 box around known best
#   F2 → UCB (κ=0.8)   : Exploit confirmed best at (0.697, 0.321)
#   F3 → UCB (κ=2.0)   : Moderate exploration
#   F4 → EI            : Exploit W11 breakthrough at (0.418, 0.384, 0.408, 0.350)
#   F5 → Override      : Hard boundary override to (0.980, 0.980, 0.990, 0.990)
#   F6 → EI            : Exploit confirmed best at (0.478, 0.319, 0.544, 0.783, 0.101)
#   F7 → EI            : Consistent improver — unchanged throughout
#   F8 → TuRBO         : Trust-region BO for 8D — global GP unreliable
#
# REFERENCES:
#   [1] Hennig & Schuler (2012) — Entropy Search for Information-Efficient
#       Global Optimization. JMLR 13, 1809-1837.
#   [2] Eriksson et al. (2019) — Scalable Global Optimization via Local
#       Bayesian Optimization (TuRBO). NeurIPS 2019.
#   [3] Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning.
#   [4] OpenML benchmark datasets — contextualising F7/F8 HP tuning outputs.
#
# AUTHOR:  BBO Capstone Project
# CONTACT: abilsilv@ic.ac.uk
# REPO:    https://github.com/andreia-bs/Capstone-project
# =============================================================================

import sys
import numpy as np
np.set_printoptions(linewidth=np.inf)

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')   # fallback for headless environments
import matplotlib.pyplot as plt

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval
from botorch.generation import MaxPosteriorSampling


# =============================================================================
# ── CONFIGURATION — EDIT THESE VALUES BEFORE RUNNING ─────────────────────────
# =============================================================================

WEEK_NO     = 13   # Number of weekly queries already submitted (update each round)
FUNCTION_NO = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 1–8

# All-time best outputs (for results summary — update after each round)
ALL_TIME_BESTS = {
    1: {"output": 1.81e-9,   "input": [0.703500, 0.716000],                                          "week": 2},
    2: {"output": 0.8135,    "input": [0.701523, 0.379257],                                          "week": 13},
    3: {"output": -0.03484,  "input": [0.492581, 0.611593, 0.340176],                                "week": 0},
    4: {"output": 0.56613,   "input": [0.418286, 0.384490, 0.407242, 0.350393],                      "week": 11},
    5: {"output": 8662.48,   "input": [1.0, 1.0, 1.0, 1.0],                                          "week": 8},
    6: {"output": -0.10788,  "input": [0.442035, 0.373720, 0.624334, 0.734770, 0.090751],            "week": 13},
    7: {"output": 3.0560,    "input": [0.174678, 0.000000, 0.504524, 0.280666, 0.306003, 0.661563],  "week": 13},
    8: {"output": 9.9668,    "input": [0.120073, 0.227667, 0.153431, 0.175346,
                                        0.955445, 0.396587, 0.208539, 0.708051],                      "week": 12},
}


# =============================================================================
# ── PER-FUNCTION KERNEL CONFIGURATION ────────────────────────────────────────
# =============================================================================
# Length-scale bounds control how smooth the GP assumes the function to be.
#   - Tight bounds (e.g. 0.005–0.05): GP is sensitive to sharp local changes
#   - Wide bounds  (e.g. 0.1–1.5):   GP assumes broad, smooth variation
#
# ν (nu) controls Matérn kernel differentiability:
#   ν=1.5 → once-differentiable  (jagged/sharp functions: F1, F2, F4)
#   ν=2.5 → twice-differentiable (smooth functions: F3, F5, F6, F7, F8)
#
# ARD (Automatic Relevance Determination) is enabled for all functions with
# N>1 — the GP learns a separate length-scale per input dimension, allowing
# it to automatically down-weight irrelevant dimensions.
# =============================================================================

LENGTH_SCALE_BOUNDS = {
    1: (0.005, 0.05),  # Very tight — near-singular sharp peak
    2: (0.05,  0.5),   # Moderate — 2D log-likelihood surface
    3: (0.1,   1.0),   # Smooth — 3D drug side-effect function
    4: (0.05,  0.5),   # Tighter — jagged 4D warehouse optimisation
    5: (0.1,   1.5),   # Wide — smooth 4D chemical yield (unimodal)
    6: (0.05,  0.75),  # Moderate — 5D cake recipe optimisation
    7: (0.05,  1.0),   # ARD important — 6D hyperparameter tuning [Ref 4]
    8: (0.05,  1.0),   # TuRBO manages local scales — 8D HP tuning [Ref 2]
}[FUNCTION_NO]

NU_SMOOTHNESS = {
    1: 1.5,   # Sharp/noisy contamination peak
    2: 1.5,   # Moderate sharpness
    3: 2.5,   # Smooth drug side-effect surface
    4: 1.5,   # Jagged warehouse optimisation landscape
    5: 2.5,   # Smooth unimodal chemical yield
    6: 2.5,   # Smooth cake recipe penalty surface
    7: 2.5,   # Smooth HP tuning response surface
    8: 2.5,   # Smooth 8D surface; TuRBO handles locality [Ref 2]
}[FUNCTION_NO]


# =============================================================================
# ── PER-FUNCTION ACQUISITION STRATEGY ────────────────────────────────────────
# =============================================================================
# Each function uses the acquisition strategy best suited to its observed
# behaviour after 12 rounds of querying.
#
# Strategy options:
#   'localised_ei' → EI constrained to small box around known best (F1)
#   'ucb'          → Upper Confidence Bound with function-specific κ (F2, F3)
#   'ei'           → Expected Improvement — gold standard for smooth fns (F4,F6,F7)
#   'override'     → Hard-coded point — bypasses GP entirely (F5)
#   'turbo'        → Trust-Region BO for high-dimensional functions (F8)
# =============================================================================

ACQUISITION_STRATEGY = {
    1: 'localised_ei',  # EI in ±0.010 box around (0.703500, 0.716000)
                        # Global EI failed for 10 rounds; hard override = W2 duplicate
    2: 'ucb',           # κ=0.8 — exploit confirmed best at (0.697, 0.321)
    3: 'ucb',           # κ=2.0 — moderate exploration of 3D space
    4: 'ei',            # Exploit W11 breakthrough (Thompson Sampling discovered it)
    5: 'override',      # True optimum at (1,1,1,1) confirmed; probe boundary neighbourhood
    6: 'ei',            # EI replaces Thompson Sampling after 3 consecutive regressions
    7: 'ei',            # Consistent improver across 6 of 12 rounds — never changed
    8: 'turbo',         # TuRBO essential for 8D — global GP unreliable [Ref 2]
}[FUNCTION_NO]

# UCB exploration parameter κ — higher = more exploration, lower = more exploitation
# Only used when ACQUISITION_STRATEGY == 'ucb'
UCB_KAPPA = {
    2: 0.8,   # Reduced from 3.0 → 1.0 → 0.8 over successive rounds
    3: 2.0,   # Moderate — F3 landscape still partially unexplored
}


# =============================================================================
# ── MANUAL OVERRIDES ──────────────────────────────────────────────────────────
# =============================================================================

# F1: Localised search box — ±0.010 around the known best (0.703500, 0.716000)
# The hard override to this exact point would duplicate Week 2.
# EI within this box is guaranteed to return a different point because
# EI = 0 exactly at all previously observed points by construction.
F1_LOCALISED_BOUNDS = torch.tensor(
    [[0.6935, 0.7060],   # lower bounds: [x1_lb, x2_lb]
     [0.7135, 0.7260]],  # upper bounds: [x1_ub, x2_ub]
    dtype=torch.float64
)

# F5: Boundary neighbourhood override
# Confirmed optimum: (1,1,1,1) = 8662.48 (Week 8) — cannot re-query
# (0.99,0.99,0.99,0.99) = 7915.74 already submitted in Week 10
# This point probes a new untested near-boundary combination
F5_BOUNDARY_OVERRIDE = np.array([0.980, 0.980, 0.990, 0.990])


# =============================================================================
# ── TURBO HYPERPARAMETERS (F8 only) ──────────────────────────────────────────
# =============================================================================
# TuRBO maintains a trust region (TR) of side length L around the current
# best point. Queries are drawn only within the TR, preventing the GP from
# being queried in regions where it has no data — a key failure mode of
# standard BO in 8D. [Ref 2]
#
# The TR adapts dynamically:
#   - Expands (×2.0) after TURBO_SUCCESS_TOL consecutive improvements
#   - Shrinks (×0.5) after TURBO_FAILURE_TOL consecutive failures
#   - Resets to initial length if it falls below TURBO_LENGTH_MIN
# =============================================================================

TURBO_LENGTH_INIT    = 0.4   # Initial TR side length (fraction of [0,1]^8)
TURBO_LENGTH_MIN     = 0.01  # Minimum TR length before reset
TURBO_LENGTH_MAX     = 1.6   # Maximum TR length
TURBO_SUCCESS_TOL    = 3     # Consecutive successes before expanding TR
TURBO_FAILURE_TOL    = 4     # Consecutive failures before shrinking TR
TURBO_SUCCESS_FACTOR = 2.0   # TR expansion multiplier
TURBO_FAILURE_FACTOR = 0.5   # TR contraction multiplier


# =============================================================================
# ── NOISE FLOOR CONFIGURATION ─────────────────────────────────────────────────
# =============================================================================
# The noise constraint prevents the GP from absorbing genuine signal as
# observation noise, which would flatten the posterior and suppress
# acquisition peaks. Tighter constraints are used for functions that
# appear deterministic and sharp (F1, F4).
# =============================================================================

NOISE_CONSTRAINT = {
    1: Interval(1e-6, 1e-3),   # Very tight — prevent noise absorption on sharp F1
    2: Interval(1e-5, 1e-1),   # Moderate
    3: Interval(1e-5, 1e-1),   # Moderate
    4: Interval(1e-5, 1e-2),   # Tight — jagged F4 landscape
    5: Interval(1e-5, 1e-1),   # Moderate — smooth unimodal
    6: Interval(1e-5, 1e-1),   # Moderate
    7: Interval(1e-5, 1e-1),   # Moderate — HP tuning outputs can be noisy
    8: Interval(1e-5, 1e-1),   # Moderate
}[FUNCTION_NO]


# =============================================================================
# DATA LOADING
# =============================================================================

def read_data(debug=False, plot_datapoints=True):
    """
    Load the initial dataset and all weekly query results for the selected function.

    File structure expected:
        ./initial_data/function_{N}/initial_inputs.npy   — shape (n_initial, N_dims)
        ./initial_data/function_{N}/initial_outputs.npy  — shape (n_initial,)
        ./week{k}_inputs.txt   — line {FUNCTION_NO-1} contains hyphen-separated inputs
        ./week{k}_outputs.txt  — line {FUNCTION_NO-1} contains the scalar output

    Parameters
    ----------
    debug : bool
        If True, prints all data points to stdout.
    plot_datapoints : bool
        If True, displays a scatter plot of all outputs coloured by value.

    Returns
    -------
    X : np.ndarray, shape (n_total, N_dims)
        All input points (initial + weekly queries).
    y : np.ndarray, shape (n_total,)
        All output values.
    """
    # Load initial dataset provided at the start of the capstone
    X = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_inputs.npy")
    y = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_outputs.npy")

    # Append each weekly query result
    for week in range(1, WEEK_NO):
        with open(f"./week{week}_inputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_x = np.array([float(v) for v in line.strip().split('-')])
            X = np.vstack((X, new_x))
        with open(f"./week{week}_outputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO - 1]
            new_y = float(line.strip())
            y = np.append(y, new_y)

    if debug:
        print(f"\n--- Data for Function {FUNCTION_NO} ({X.shape[0]} points) ---")
        for xi, yi in zip(X, y):
            print(f"  x={xi}  →  y={yi:.6e}")

    if plot_datapoints:
        _plot_outputs(y)

    return X, y


def _plot_outputs(y):
    """Plot all output values coloured by magnitude."""
    n_initial = len(y) - (WEEK_NO - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(n_initial), y[:n_initial],
               c=y[:n_initial], vmin=y.min(), vmax=y.max(),
               marker='o', cmap='rainbow', s=50, edgecolor='black',
               label='Initial data')
    if WEEK_NO > 1:
        ax.scatter(range(n_initial, len(y)), y[n_initial:],
                   c=y[n_initial:], vmin=y.min(), vmax=y.max(),
                   marker='s', cmap='rainbow', s=50, edgecolor='black',
                   label='Weekly queries')
    ax.set_ylabel("Output value")
    ax.set_xlabel("Data point index")
    ax.set_title(f"Function {FUNCTION_NO} — all outputs (Week {WEEK_NO})")
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# GP CONSTRUCTION (shared by BoTorchGP and TuRBoGP)
# =============================================================================

def build_gp(X_np, y_np):
    """
    Construct a BoTorch SingleTaskGP with the per-function kernel configuration.

    Architecture:
        - Matérn kernel with ARD (one length-scale per input dimension)
        - ScaleKernel wrapper (learns output variance σ_f²)
        - Standardize outcome transform (zero-mean normalisation)
        - Per-function noise constraint (prevents noise absorption)

    The Standardize transform is critical: without it, the GP's prior mean
    is the raw output mean rather than zero. For functions with all-negative
    outputs (F3, F6), this would bias UCB/EI toward already-explored areas.

    Parameters
    ----------
    X_np : np.ndarray, shape (n, d)
    y_np : np.ndarray, shape (n,)

    Returns
    -------
    gp : SingleTaskGP (unfitted)
    """
    matern = MaternKernel(
        nu=NU_SMOOTHNESS,
        ard_num_dims=X_np.shape[1],
        lengthscale_constraint=Interval(*LENGTH_SCALE_BOUNDS)
    )
    covar_module = ScaleKernel(matern)

    gp = SingleTaskGP(
        train_X=torch.tensor(X_np, dtype=torch.float64),
        train_Y=torch.tensor(y_np, dtype=torch.float64).unsqueeze(-1),
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
        train_Yvar=None,
    )
    # Apply per-function noise constraint
    gp.likelihood.noise_covar.register_constraint("raw_noise", NOISE_CONSTRAINT)
    return gp


# =============================================================================
# BoTorchGP — used for Functions 1–7
# =============================================================================

class BoTorchGP:
    """
    Standard Gaussian Process surrogate with function-specific acquisition.

    Supports four acquisition strategies:
        - 'localised_ei' : EI optimised within a constrained local box (F1)
        - 'ucb'          : Upper Confidence Bound with tunable κ (F2, F3)
        - 'ei'           : Expected Improvement (F4, F6, F7)
        - 'override'     : Hard-coded point, GP bypassed entirely (F5)
    """

    def fit(self, X, y, debug=False):
        """
        Fit the GP surrogate by maximising the marginal log-likelihood (MLL).

        MLL fitting optimises the kernel hyperparameters (length-scales,
        output variance, noise) to best explain the observed data. Multiple
        restarts are used internally by fit_gpytorch_mll to avoid local optima
        in the MLL landscape.
        """
        self.N    = X.shape[1]
        self.X_np = X
        self.y_np = y

        self.gp = build_gp(X, y)
        self.gp.train()
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        self.gp.eval()

        if debug:
            print(f"\n--- BoTorchGP fitted [{ACQUISITION_STRATEGY.upper()}] ---")
            print(f"  σ_f² (output scale) : {self.gp.covar_module.outputscale.item():.4f}")
            ls = self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
            print(f"  ℓ   (length scales) : {ls}")
            print(f"  σ_n² (noise)        : {self.gp.likelihood.noise.item():.2e}")

    def evaluate(self, X, grid=False):
        """
        Evaluate the GP posterior mean and standard deviation at new points.

        Parameters
        ----------
        X : np.ndarray or list of meshgrid arrays
        grid : bool — if True, X is a meshgrid; flatten then reshape output

        Returns
        -------
        mean : np.ndarray — posterior mean
        std  : np.ndarray — posterior standard deviation (sqrt of variance)
        """
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            post = self.gp.posterior(torch.tensor(X, dtype=torch.float64))
            mean = post.mean.detach().numpy().ravel()
            std  = post.variance.sqrt().detach().numpy().ravel()
        if grid:
            mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std

    def _build_acquisition(self):
        """
        Instantiate the acquisition function for the current strategy.

        EI (Expected Improvement):
            Computes E[max(f(x) - f*, 0)] — the expected amount by which
            the next query will exceed the current best f*. Naturally
            balances exploration and exploitation without a tunable parameter.

        UCB (Upper Confidence Bound):
            μ(x) + κ·σ(x) — trades off mean prediction against uncertainty.
            Higher κ → more exploration; lower κ → more exploitation.
        """
        if ACQUISITION_STRATEGY in ('ei', 'localised_ei', 'override'):
            best_f = torch.tensor(float(self.y_np.max()), dtype=torch.float64)
            return ExpectedImprovement(model=self.gp, best_f=best_f)

        elif ACQUISITION_STRATEGY == 'ucb':
            kappa = UCB_KAPPA[FUNCTION_NO]
            return UpperConfidenceBound(model=self.gp, beta=kappa ** 2)

        else:
            raise ValueError(f"Unknown strategy: {ACQUISITION_STRATEGY}")

    def evaluate_acquisition(self, X, grid=False):
        """Evaluate the acquisition function at a set of candidate points."""
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        acq = self._build_acquisition()
        with torch.no_grad():
            vals = acq(torch.tensor(X, dtype=torch.float64).unsqueeze(1))
            vals = vals.detach().numpy().ravel()
        if grid:
            vals = vals.reshape(*shape)
        return vals

    def find_acquisition_max(self, debug=False):
        """
        Find the input point that maximises the acquisition function.

        Three dispatch paths:
        1. F1 (localised_ei): EI optimised within F1_LOCALISED_BOUNDS
           Guarantees a new point — EI=0 at all observed points by construction
        2. F5 (override): Returns F5_BOUNDARY_OVERRIDE directly, GP bypassed
        3. All others (ei, ucb): Global optimize_acqf over [0,1]^N

        Returns
        -------
        best_x : np.ndarray — proposed query point
        best_y : float      — acquisition value at best_x
        """

        # ── F1: Localised EI ──────────────────────────────────────────────────
        if FUNCTION_NO == 1:
            acq = self._build_acquisition()
            best_x, best_y = optimize_acqf(
                acq_function=acq,
                bounds=F1_LOCALISED_BOUNDS,   # constrained ±0.010 box
                raw_samples=512,
                num_restarts=32,
                q=1
            )
            best_x = best_x.detach().numpy().ravel()
            best_y = best_y.detach().numpy().item()
            if debug:
                print("\n--- F1: Localised EI ---")
                print("  Search box: x1 ∈ [0.6935, 0.7135], x2 ∈ [0.7060, 0.7260]")
                print(f"  Proposed  : {'-'.join(f'{v:.6f}' for v in best_x)}")
                print(f"  EI value  : {best_y:.4e}")
                print("  Note: guaranteed ≠ (0.703500, 0.716000) by EI construction")
            return best_x, best_y

        # ── F5: Hard boundary override ────────────────────────────────────────
        if FUNCTION_NO == 5:
            best_x = F5_BOUNDARY_OVERRIDE
            mean, _ = self.evaluate(best_x.reshape(1, -1))
            best_y  = float(mean[0])
            if debug:
                print("\n--- F5: Boundary Override ---")
                print(f"  Override  : {'-'.join(f'{v:.6f}' for v in best_x)}")
                print(f"  GP mean   : {best_y:.4f}")
            return best_x, best_y

        # ── All other functions: global optimisation ──────────────────────────
        bounds = torch.stack([
            torch.zeros(self.N, dtype=torch.float64),
            torch.ones(self.N,  dtype=torch.float64)
        ])
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
            print(f"  Proposed  : {'-'.join(f'{v:.6f}' for v in best_x)}")
            print(f"  Acq value : {best_y:.4e}")

        return best_x, best_y


# =============================================================================
# TuRBoGP — used exclusively for Function 8
# =============================================================================

class TuRBoGP:
    """
    Trust-Region Bayesian Optimisation for high-dimensional Function 8.

    Standard global BO fails in 8D because:
      (a) The GP covariance matrix becomes ill-conditioned
      (b) The acquisition function has exponentially many local maxima
      (c) The GP posterior is overconfident in unexplored regions

    TuRBO [Ref 2] addresses all three by restricting queries to a trust
    region (TR) centred on the current best point. The TR is anisotropic —
    wider in dimensions with longer length-scales (smoother, less sensitive)
    and narrower in dimensions with shorter length-scales (sharper, more
    sensitive). Thompson Sampling is used within the TR.
    """

    def __init__(self):
        self.tr_length       = TURBO_LENGTH_INIT
        self.success_counter = 0
        self.failure_counter = 0
        self.best_y          = -np.inf

    def fit(self, X, y, debug=False):
        """Fit the GP on all available data. TR restricts where we query, not how we fit."""
        self.N      = X.shape[1]
        self.X_np   = X
        self.y_np   = y
        self.best_y = float(y.max())

        self.gp = build_gp(X, y)
        self.gp.train()
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        self.gp.eval()

        if debug:
            print("\n--- TuRBoGP fitted [TuRBO + Thompson Sampling] ---")
            print(f"  σ_f² (output scale) : {self.gp.covar_module.outputscale.item():.4f}")
            ls = self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
            print(f"  ℓ   (length scales) : {ls}")
            print(f"  σ_n² (noise)        : {self.gp.likelihood.noise.item():.2e}")
            print(f"  Trust region length : {self.tr_length:.4f}")
            print(f"  Current best y      : {self.best_y:.6f}")

    def _get_tr_bounds(self):
        """
        Compute anisotropic trust region bounds centred on the current best point.

        The TR half-width in each dimension is scaled by the fitted length-scale:
            half_width_d = (tr_length / 2) × (ℓ_d / max(ℓ))
        Dimensions with longer length-scales (smoother) get wider TR bounds.
        """
        best_idx   = np.argmax(self.y_np)
        x_center   = self.X_np[best_idx]
        ls         = self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        weights    = ls / ls.max()           # normalise: longest dim gets full width
        half_width = (self.tr_length / 2.0) * weights
        lb = np.clip(x_center - half_width, 0.0, 1.0)
        ub = np.clip(x_center + half_width, 0.0, 1.0)
        return lb, ub

    def update_tr_state(self, new_y):
        """
        Update TR length after observing a new output.
        Call this after each weekly result is received.
        """
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
        """Evaluate GP posterior mean and std at candidate points."""
        if grid:
            shape = X[0].shape
            X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            post = self.gp.posterior(torch.tensor(X, dtype=torch.float64))
            mean = post.mean.detach().numpy().ravel()
            std  = post.variance.sqrt().detach().numpy().ravel()
        if grid:
            mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std

    def evaluate_acquisition(self, X, grid=False):
        """
        UCB proxy (κ=2) masked to zero outside the trust region.
        Used for visualisation only — actual query uses Thompson Sampling.
        """
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
        """
        Propose the next query using Thompson Sampling within the trust region.

        Thompson Sampling draws a random sample path from the GP posterior
        and returns the candidate with the highest sampled value. Restricting
        candidates to the TR ensures the GP is only queried where it has
        sufficient data to be reliable.
        """
        lb, ub = self._get_tr_bounds()
        lb_t = torch.tensor(lb, dtype=torch.float64)
        ub_t = torch.tensor(ub, dtype=torch.float64)

        # Draw 2048 uniform candidates within the trust region
        candidates = lb_t + (ub_t - lb_t) * torch.rand(2048, self.N, dtype=torch.float64)

        ts = MaxPosteriorSampling(model=self.gp, replacement=False)
        with torch.no_grad():
            best_x = ts(candidates, num_samples=1).detach().numpy().ravel()

        mean, _ = self.evaluate(best_x.reshape(1, -1))
        best_y  = float(mean[0])

        if debug:
            print("\n--- TuRBO + Thompson Sampling ---")
            print(f"  Trust region length : {self.tr_length:.4f}")
            print(f"  TR lower bound      : {lb}")
            print(f"  TR upper bound      : {ub}")
            print(f"  Proposed            : {'-'.join(f'{v:.6f}' for v in best_x)}")
            print(f"  GP mean at proposal : {best_y:.4f}")

        return best_x, best_y


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_gp_results(model, X, y):
    """
    Visualise the GP posterior and acquisition function.

    Plot type is selected automatically based on input dimensionality:
        N=1 : 2D line plot (posterior mean ± std + acquisition)
        N=2 : 3D surface plot
        N=3 : 3D scatter with acquisition colour-map
        N>3 : Parallel coordinates plot (one line per data point)
    """
    N = X.shape[1]
    acq_label = ACQUISITION_STRATEGY.upper()

    if N == 1:
        xs = np.linspace(0, 1, 100).reshape(-1, 1)
        mean, std = model.evaluate(xs, grid=False)
        acq_vals  = model.evaluate_acquisition(xs, grid=False)
        best_x, best_y_acq = model.find_acquisition_max()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X[:-WEEK_NO+1], y[:-WEEK_NO+1], color='black', marker='o', s=40, label='Initial data')
        ax.scatter(X[-WEEK_NO+1:], y[-WEEK_NO+1:], color='red',   marker='s', s=60, label='Weekly queries')
        ax.plot(xs, mean, linewidth=2, label='GP posterior mean')
        ax.fill_between(xs.ravel(), mean-std, mean+std, alpha=0.2, label='±1 std')
        ax.plot(xs, acq_vals, color='green', linewidth=2, linestyle='--', label=f'Acquisition [{acq_label}]')
        ax.scatter(best_x, best_y_acq, marker='*', s=300, color='green', zorder=5, label='Proposed query')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'GP Posterior — Function {FUNCTION_NO} [{acq_label}]')
        ax.legend(); plt.tight_layout(); plt.show()

    elif N == 2:
        ls = np.linspace(0, 1, 25)
        mesh = np.meshgrid(ls, ls, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        best_x, _ = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:-WEEK_NO+1,0], X[:-WEEK_NO+1,1], y[:-WEEK_NO+1], marker='o', color='black', s=40)
        ax.scatter(X[-WEEK_NO+1:,0], X[-WEEK_NO+1:,1], y[-WEEK_NO+1:], marker='s', color='red',   s=60)
        ax.plot_surface(mesh[0], mesh[1], mean, cmap='rainbow', alpha=0.8, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], mean+std, color='lightgray', alpha=0.15, linewidth=0)
        ax.plot_surface(mesh[0], mesh[1], mean-std, color='lightgray', alpha=0.15, linewidth=0)
        ax.scatter(best_x[0], best_x[1], model.evaluate(best_x.reshape(1,-1))[0][0],
                   color='green', s=200, marker='*', label='Proposed query')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
        ax.set_title(f'GP Posterior — Function {FUNCTION_NO} [{acq_label}]')
        plt.tight_layout(); plt.show()

    elif N == 3:
        ls = np.linspace(0, 1, 15)
        mesh = np.meshgrid(ls, ls, ls, indexing='ij')
        acq_vals = model.evaluate_acquisition(mesh, grid=True)
        best_x, _ = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        sizes = 40 * (y - y.min()) / (y.max() - y.min() + 1e-9) + 10
        ax.scatter(X[:-WEEK_NO+1,0], X[:-WEEK_NO+1,1], X[:-WEEK_NO+1,2],
                   s=sizes[:-WEEK_NO+1], c=y[:-WEEK_NO+1], cmap='rainbow', marker='o')
        ax.scatter(X[-WEEK_NO+1:,0], X[-WEEK_NO+1:,1], X[-WEEK_NO+1:,2],
                   s=sizes[-WEEK_NO+1:], c=y[-WEEK_NO+1:], cmap='rainbow', marker='s')
        ax.scatter(best_x[0], best_x[1], best_x[2], color='green', s=300, marker='*')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
        ax.set_title(f'Observed Data — Function {FUNCTION_NO} [{acq_label}]')
        plt.tight_layout(); plt.show()

    else:  # N > 3: parallel coordinates
        best_x, _ = model.find_acquisition_max()
        fig, ax = plt.subplots(figsize=(12, 6))
        norm = plt.Normalize(y.min(), y.max())
        cmap = plt.cm.rainbow
        x_axis = np.arange(N)

        for i in range(len(X) - WEEK_NO + 1):
            ax.plot(x_axis, X[i], color=cmap(norm(y[i])), linewidth=1, alpha=0.5)
        for i in range(len(X) - WEEK_NO + 1, len(X)):
            ax.plot(x_axis, X[i], color=cmap(norm(y[i])), linewidth=2, linestyle='--', alpha=0.8)
        ax.plot(x_axis, best_x, color='green', linewidth=3, linestyle='--', label='Proposed query')

        ax.set_xticks(x_axis)
        ax.set_xticklabels([f'x[{d+1}]' for d in range(N)])
        ax.set_ylabel('Input value')
        ax.set_title(f'Parallel Coordinates — Function {FUNCTION_NO} [{acq_label}]')
        ax.legend()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Output value')
        plt.tight_layout(); plt.show()


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(best_x):
    """Print a formatted summary of the proposed query and all-time bests."""
    print("\n" + "="*65)
    print(f"  FUNCTION {FUNCTION_NO} — PROPOSED QUERY (Week {WEEK_NO})")
    print("="*65)
    print(f"  Strategy  : {ACQUISITION_STRATEGY.upper()}")
    print(f"  Query     : {'-'.join(f'{v:.6f}' for v in best_x)}")
    print(f"\n  ✅ PORTAL SUBMISSION STRING:")
    print(f"     {'-'.join(f'{v:.6f}' for v in best_x)}")

    atb = ALL_TIME_BESTS[FUNCTION_NO]
    print(f"\n  All-time best output : {atb['output']:.6e}")
    print(f"  All-time best input  : {'-'.join(f'{v:.6f}' for v in atb['input'])}")
    print(f"  Achieved in          : Week {atb['week']}")
    print("="*65)

    print("\n  FULL CAPSTONE SUMMARY")
    print("  " + "-"*60)
    print(f"  {'Fn':<4} {'All-Time Best':>16} {'Week':>6}  Input")
    print("  " + "-"*60)
    for fn, info in ALL_TIME_BESTS.items():
        inp_str = '-'.join(f'{v:.4f}' for v in info['input'])
        marker = " ◄" if fn == FUNCTION_NO else ""
        print(f"  F{fn:<3} {info['output']:>16.6e} {info['week']:>6}  {inp_str}{marker}")
    print("  " + "-"*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main execution pipeline:
        1. Load all accumulated data
        2. Fit the GP surrogate
        3. Find the acquisition maximum (proposed next query)
        4. Visualise the GP posterior
        5. Print the portal submission string
    """
    print(f"\n{'='*65}")
    print(f"  BBO Capstone — Function {FUNCTION_NO} | Week {WEEK_NO}")
    print(f"  Strategy: {ACQUISITION_STRATEGY.upper()}")
    print(f"{'='*65}")

    # Step 1: Load data
    X, y = read_data(debug=False, plot_datapoints=True)
    print(f"\n  Data loaded: {X.shape[0]} points, {X.shape[1]} dimensions")
    print(f"  Output range: [{y.min():.4e}, {y.max():.4e}]")

    # Step 2: Fit surrogate
    model = TuRBoGP() if ACQUISITION_STRATEGY == 'turbo' else BoTorchGP()
    model.fit(X, y, debug=True)

    # Step 3: Find next query
    best_x, best_acq = model.find_acquisition_max(debug=True)

    # Step 4: Visualise
    plot_gp_results(model, X, y)

    # Step 5: Print results
    print_results_summary(best_x)


if __name__ == '__main__':
    main()
