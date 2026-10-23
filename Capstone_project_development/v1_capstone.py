import math
import sys

import numpy as np
np.set_printoptions(linewidth=np.inf)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import warnings
from sklearn.exceptions import ConvergenceWarning

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import Interval

WEEK_NO = 6

# Select the function number from command line argument
FUNCTION_NO = int(sys.argv[1]) if len(sys.argv) > 1 else 1

LENGTH_SCALE_BOUNDS = {
    1: (0.02, 0.3),
    2: (0.05, 0.5),
    3: (0.1, 1.0),
    4: (0.05, 0.5),
    5: (0.1, 1.5),
    6: (0.05, 0.75),
    7: (0.05, 1.0),
    8: (0.05, 1.0)
}[FUNCTION_NO]

NU_SMOOTHNESS = {
    1: 1.5,
    2: 1.5,
    3: 2.5,
    4: 1.5,
    5: 2.5,
    6: 2.5,
    7: 2.5,
    8: 2.5
}[FUNCTION_NO]

# The UCB acquisition function is defined as: UCB(x) = post_mean(x) + kappa * post_std(x)
# Kappa is a parameter that controls the trade-off between exploration and exploitation.
# Lower values of kappa make the acquisition function more exploitative, while higher values make it more exploratory.
UCB_KAPPA = {
    1: 3,
    2: 3,
    3: 2,
    4: 3,
    5: 1000,
    6: 2.5,
    7: 2,
    8: 2.5
}[FUNCTION_NO]


def main():

    # --- Read data from files ---
    X, y = read_data(debug=False, plot_datapoints=False)
    N = X.shape[1]

    # --- Use the Gaussian Process model from SciKit ---
    # scikit_gp = SciKitGP()
    # scikit_gp.fit(X, y, debug=True)
    # scikit_gp.find_acquisition_max(debug=True)
    # plot_gp_results(scikit_gp, X, y)

    # --- Use the Gaussian Process model from BoTorch ---
    botorch_gp = BoTorchGP()
    botorch_gp.fit(X, y, debug=True)
    botorch_gp.find_acquisition_max(debug=True)
    plot_gp_results(botorch_gp, X, y)


# --- Read data from files and plot outputs ---
def read_data(debug=False, plot_datapoints=True):

    # --- Load the initial data ---
    X = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_inputs.npy")
    y = np.load(f"./initial_data/function_{FUNCTION_NO}/initial_outputs.npy")
    N = X.shape[1]

    # --- Add data from previous weeks ---
    for week in range(1, WEEK_NO):
        with open(f"./week{week}_inputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO-1]
            new_values = np.array([float(x) for x in line.strip().split('-')])
            X = np.vstack((X, new_values))
        with open(f"./week{week}_outputs.txt", 'r') as f:
            line = f.readlines()[FUNCTION_NO-1]
            new_value = float(line.strip())
            y = np.append(y, np.array(new_value))

    # --- Print the inputs and outputs for the selected function ---
    if debug:
        print("\n--- Data points ---")
        for Xi, yi in zip(X, y):
            print(Xi, yi)

    # --- Plot the outputs with colors ---
    if plot_datapoints:
        sorted_indices = np.argsort(y)
        sorted_indices = np.arange(len(y)) # Uncomment this line to disable sorting and keep the original order of the data
        rank_indices = np.argsort(sorted_indices)
        previous_data_indices = rank_indices[:-WEEK_NO+1]
        latest_data_indices = rank_indices[-WEEK_NO+1:]
        sorted_y = y[sorted_indices]

        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
        ax.scatter(previous_data_indices, sorted_y[previous_data_indices], c=sorted_y[previous_data_indices], vmin=y.min(), vmax=y.max(), marker='o', cmap='rainbow', s=50, edgecolor='black')
        ax.scatter(latest_data_indices, sorted_y[latest_data_indices], c=sorted_y[latest_data_indices], vmin=y.min(), vmax=y.max(), marker='s', cmap='rainbow', s=50, edgecolor='black')
        ax.set_ylabel("y")
        ax.set_title(f'Outputs for function {FUNCTION_NO}')
        plt.show()

        # --- Plot the outputs for each input dimension separately ---
        if N > 1:
            nrows = int(np.ceil(np.sqrt(N)))
            ncols = int(np.ceil(N / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3*nrows))
            axes = axes.flatten()
            for i in range(N):
                axes[i].scatter(X[:-WEEK_NO+1, i], y[:-WEEK_NO+1], c=y[:-WEEK_NO+1], vmin=y.min(), vmax=y.max(), marker='o', cmap='rainbow', s=50, edgecolor='black')
                axes[i].scatter(X[-WEEK_NO+1:, i], y[-WEEK_NO+1:], c=y[-WEEK_NO+1:], vmin=y.min(), vmax=y.max(), marker='s', cmap='rainbow', s=50, edgecolor='black')
                axes[i].set_xlabel(f"x[{i+1}]")
                axes[i].set_ylabel("y")
                axes[i].set_title(f'Inputs for dimension {i+1} of function {FUNCTION_NO}')
            plt.tight_layout()
            plt.show()

    return X, y


# --- Plot the GP and acquisition function results ---
def plot_gp_results(model, X, y):
    
    N = X.shape[1]

    # --- Plotting for 2D cases ---
    if N == 1:
        linspace = np.linspace(0, 1, 100)
        mean, std = model.evaluate(linspace.reshape(-1, 1), grid=False)
        acquisition = model.evaluate_acquisition(linspace.reshape(-1, 1), grid=False)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()

        # Scatter of the actual data points
        ax.scatter(X[:-WEEK_NO+1], y[:-WEEK_NO+1], color='black', marker='o', s=40, label='Initial data')
        ax.scatter(X[-WEEK_NO+1:], y[-WEEK_NO+1:], color='black', marker='o', s=40, label='Initial data')

        # Posterior mean
        ax.plot(linspace, mean, linewidth=2, label='Posterior mean')

        # Upper and lower std bounds surfaces
        ax.fill_between(linspace, mean - std, mean + std, alpha=0.2, label='Posterior std')

        # Acquisition function
        ax.plot(linspace, acquisition, color='green', linewidth=2, label='Acquisition function')

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X, acquisition_max_y, marker='*', s=200, color='green', label='Acquisition max')

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'GP posterior for function {FUNCTION_NO}')
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

    # --- Plotting for 3D cases ---
    if N == 2:
        linspace = np.linspace(0, 1, 25)
        mesh = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter of the data points
        ax.scatter(X[:-WEEK_NO+1, 0], X[:-WEEK_NO+1, 1], y[:-WEEK_NO+1], marker='o', color='black', s=50, depthshade=True, label='Initial data')
        ax.scatter(X[-WEEK_NO+1:, 0], X[-WEEK_NO+1:, 1], y[-WEEK_NO+1:], marker='s', color='black', s=50, depthshade=True, label='Weekly data')

        # Posterior mean surface
        ax.plot_surface(mesh[0], mesh[1], mean, cmap='rainbow', alpha=0.8, linewidth=0, antialiased=True, label='Posterior mean')

        # Upper and lower std bounds surfaces
        ax.plot_surface(mesh[0], mesh[1], mean + std, color='lightgray', alpha=0.2, linewidth=0, label='Posterior std upper bound')
        ax.plot_surface(mesh[0], mesh[1], mean - std, color='lightgray', alpha=0.2, linewidth=0, label='Posterior std lower bound')

        # Acquisition function surface
        ax.plot_surface(mesh[0], mesh[1], acquisition, color='green', alpha=0.2, linewidth=0, label='Acquisition function')

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_y, color='green', s=200, marker='*', label='Acquisition max')

        # Set labels and title
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title(f'GP posterior for function {FUNCTION_NO}')
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

    # --- Plotting for 4D cases ---
    if N == 3:
        linspace = np.linspace(0, 1, 20)
        mesh = np.meshgrid(*(linspace,) * N, indexing='ij')
        mean, std = model.evaluate(mesh, grid=True)
        acquisition = model.evaluate_acquisition(mesh, grid=True)
        acquisition_max_X, acquisition_max_y = model.find_acquisition_max()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter of the actual data points with colour and size based on output value
        sizes = 40 * (y - y.min()) / (y.max() - y.min()) + 10  # Scale sizes between 10 and 50
        ax.scatter(X[:-WEEK_NO+1, 0], X[:-WEEK_NO+1, 1], X[:-WEEK_NO+1, 2], s=sizes[:-WEEK_NO+1], c=y[:-WEEK_NO+1], cmap='rainbow', vmin=y.min(), vmax=y.max(), marker='o', depthshade=True, label='Initial data')
        ax.scatter(X[-WEEK_NO+1:, 0], X[-WEEK_NO+1:, 1], X[-WEEK_NO+1:, 2], s=sizes[-WEEK_NO+1:], c=y[-WEEK_NO+1:], cmap='rainbow', vmin=y.min(), vmax=y.max(), marker='s', depthshade=True, label='Weekly data')

        # Plot the acquisition function as a scatter plot
        ax.scatter(mesh[0], mesh[1], mesh[2], c=acquisition, cmap='viridis', marker='o', s=100, alpha=0.4, edgecolors='none', label='Acquisition function')

        # Plot the acquisition function maximum point
        ax.scatter(acquisition_max_X[0], acquisition_max_X[1], acquisition_max_y, color='green', s=200, marker='*', label='Acquisition max')

        # Set labels and title
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title(f'Observed data for function {FUNCTION_NO}')
        ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

    # --- Plotting for higher-dimensional cases (5D+) ---
    if N > 3:
        acquisition_max_X, _ = model.find_acquisition_max()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize y for colormap
        norm = plt.Normalize(y.min(), y.max())
        cmap = plt.cm.rainbow

        x_axis = np.arange(N)

        # Plot each point as a line in the parallel coordinates plot
        for i in range(X.shape[0]-WEEK_NO+1):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=1, alpha=0.6)
        for i in range(X.shape[0]-WEEK_NO+1, X.shape[0]):
            ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=2, linestyle='--', alpha=0.6)

        # Plot the acquisition function maximum point as a dashed line in the parallel coordinates plot
        ax.plot(x_axis, acquisition_max_X, color='green', linewidth=3, linestyle='--', label='Acquisition max')

        # Set labels and title
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f'x[{d}]' for d in range(1, N+1)])
        ax.set_title(f'Parallel coordinates for function {FUNCTION_NO}')
        ax.set_ylabel('Input value')
        ax.legend()

        # Colorbar to show what color maps to what y value
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='y')

        # Display the plot
        plt.tight_layout()
        plt.show()


class GPModel:

    N = None

    def fit(self, X, y, debug=False):
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, X, grid=False):
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate_acquisition(self, X, grid=False):
        raise NotImplementedError("Subclasses should implement this method.")

    def find_acquisition_max(self, debug=False):
        raise NotImplementedError("Subclasses should implement this method.")

class SciKitGP(GPModel):

    # --- Fit the Gaussian Process model ---
    def fit(self, X, y, debug=False):

        self.N = X.shape[1]

        # The constant kernel allows the regressor to control the variation of the function.
        # i.e. it doesn't assume that variation is constant across the function space.
        # If not used, the RBF kernel could incorrectly be adjusted to compensate for this variation,
        # which would intefere with the length scale calculation (the smoothness of the function).
        const_kernel = ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(0.01, 100)
        )

        # The RBF kernel allows the regressor to control the smoothness of the function.
        rbf_kernel = RBF(
            # Lower length scale values make the GP more sensitive to changes in the function (i.e. sharper), while higher values make it smoother.
            # This value defines the initial length scale used by the regressor. For functions with inputs between 0 and 1, the recommended initial value is 0.2.
            # Also, because different dimensions can have different effects to the output, a list of length scales is passed,
            # one per dimension, so that the regressor that update them independently.
            # This makes this kernel an Automatic Relevance Determination (ARD) kernel.
            length_scale=np.full(X.shape[1], 0.2),
            # Allows the GP regressor to search length scales within this range
            # For functions with inputs between 0 and 1, the recommended range is 0.1-0.5.
            # Less than 0.1 is highly not recommended given the low amount of input datapoints,
            # as it would cause the regressor to quickly overfit and assume that one dimension is much more significant than another.
            # The only exception to this is function no 1, which seems to be really sharp, and would benefit from lower length scale values.
            length_scale_bounds=LENGTH_SCALE_BOUNDS
        )

        self.gp = GaussianProcessRegressor(
            kernel = const_kernel * rbf_kernel,
            # This ensures that internally, y values are normalized around 0.
            # Not doing this would cause problems since the post_mean function assumes the value 0 in unexplored regions.
            # For example, in functions with only negative values, all explored regions with maxima would have a lower value
            # than unexplored regions. Because the post std also assumes higher values in unexplored regions,
            # the UCB acquisition function would always prioritise exploration, regardless of Kappa (post_mean + K * post_std).
            normalize_y=True,
            # This allows the regressor to run multiple times, and carrying over the adjusted kernel's prameters from previous runs.
            n_restarts_optimizer=5
        )

        with warnings.catch_warnings():
            # During the GP setup, some constraints are applied to the kernel parameters, which can cause convergence warnings during fitting.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.gp.fit(X, y)

        if debug:
            print("\n--- SciKit kernel parameters after fitting ---")
            print("Constant kernel value (variation):", self.gp.kernel_.k1.constant_value)
            # Indirectly, a high length scale value means that the dimension has less significance on the output.
            print("RBF kernel length scale per dimension (smoothness):", self.gp.kernel_.k2.length_scale)

    # --- Evaluate the GP model on new data points ---
    def evaluate(self, X, grid=False):
        if grid: shape = X[0].shape; X = np.column_stack([m.ravel() for m in X])
        mean, std = self.gp.predict(X, return_std=True)
        if grid: mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std
    
    # --- Evaluate the acquisition function (Upper Confidence Bound) on new data points ---
    def evaluate_acquisition(self, X, grid=False):
        mean, std = self.evaluate(X, grid=grid)
        return mean + UCB_KAPPA * std

    # --- Find the maximum of the acquisition function (Upper Confidence Bound) ---
    def find_acquisition_max(self, debug=False):

        initial_candidates=512
        n_starts=32
    
        # Generate the starting points for the search as the best from a list of initial candidates
        candidates = np.random.uniform(0, 1, size=(max(initial_candidates, n_starts), self.N))
        candidates_acquisition_values = self.evaluate_acquisition(candidates)
        starts = candidates[np.argsort(candidates_acquisition_values)[-n_starts:]]

        # Use the starting points to find the maximum of the acquisition function using the L-BFGS-B method (based on gradient descent)
        best_x, best_y = None, np.inf
        for x in starts:
            # UCB is a maximization function, but the minimize function is a minimization technique, so we need to negate the value of UCB.
            result = minimize(lambda x: -self.evaluate_acquisition([x]), x, bounds=[(0, 1)] * self.N, method='L-BFGS-B')
            if result.fun < best_y:
                best_y = result.fun
                best_x = result.x

        if debug:
            print("\n--- SciKit acquisition function ---")
            with np.printoptions(formatter={'float': lambda x: f"{x:.6f}"}):
                print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
            print("Maximum value:", -best_y)

        return best_x, -best_y
    

class BoTorchGP(GPModel):

    # --- Fit the Gaussian Process model ---
    def fit(self, X, y, debug=False):

        self.N = X.shape[1]

        matern_kernel = MaternKernel(
            nu=NU_SMOOTHNESS,
            ard_num_dims=X.shape[1],
            lengthscale_constraint=Interval(*LENGTH_SCALE_BOUNDS)
        )

        covar_module = ScaleKernel(matern_kernel)

        self.gp = SingleTaskGP(
            torch.tensor(X, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64).unsqueeze(-1),
            covar_module=covar_module,
            outcome_transform=Standardize(m=1))

        self.gp.train()
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

        if debug:
            print("\n--- BoTorch kernel parameters after fitting ---")
            print("Scale Kernel constant (sigma_f^2):", self.gp.covar_module.outputscale.item())
            print("Matern kernel length scale per dimension (smoothness):", self.gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel())
            print("Observation noise:", self.gp.likelihood.noise.item())

    # --- Evaluate the GP model on new data points ---
    def evaluate(self, X, grid=False):
        self.gp.eval()
        if grid: shape = X[0].shape; X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            posterior = self.gp.posterior(torch.tensor(X, dtype=torch.float64))
            mean = posterior.mean.detach().numpy().ravel()
            std = posterior.variance.sqrt().detach().numpy().ravel()
        if grid: mean, std = mean.reshape(*shape), std.reshape(*shape)
        return mean, std

    ucb = None
    def init_acquisition_function(self):
        if not self.ucb:
            self.ucb = UpperConfidenceBound(model=self.gp, beta=UCB_KAPPA**2) # Beta is equal to Kappa^2
        return self.ucb

    # --- Evaluate the acquisition function (Upper Confidence Bound) on new data points ---
    def evaluate_acquisition(self, X, grid=False):
        self.init_acquisition_function()
        if grid: shape = X[0].shape; X = np.column_stack([m.ravel() for m in X])
        with torch.no_grad():
            y = self.ucb(torch.tensor(X, dtype=torch.float64).unsqueeze(1)).detach().numpy().ravel()
        if grid: y = y.reshape(*shape)
        return y

    # --- Find the maximum of the acquisition function (Upper Confidence Bound) ---
    def find_acquisition_max(self, debug=False):

        initial_candidates=512
        n_starts=32

        best_x, best_y = optimize_acqf(
            acq_function=self.init_acquisition_function(),
            bounds=torch.stack([torch.zeros(self.N, dtype=torch.float64), torch.ones(self.N, dtype=torch.float64)]),
            raw_samples=initial_candidates,
            num_restarts=n_starts,
            q=1  # number of points to propose (1 = sequential BO)
        )
        best_x, best_y = best_x.detach().numpy().ravel(), best_y.detach().numpy().item()

        if debug:
            print("\n--- BoTorch acquisition function ---")
            with np.printoptions(formatter={'float': lambda x: f"{x:.6f}"}):
                print("Maximum at:", '-'.join(f'{x:.6f}' for x in best_x))
            print("Maximum value:", best_y)

        return best_x, best_y

if __name__ == '__main__':
    main()