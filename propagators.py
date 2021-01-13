#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np

np.seterr(over="warn", under="warn")
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from copy import deepcopy
from functools import partial
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
from numpy.matlib import repmat

from visualization import (
    grid_wrap_nrow,
    gridspec_kw,
    figsize,
    cmap_grid_code,
    cmap_state_density,
    cmap_activation_prob,
    font_scale,
    page_width,
    row_height,
    color_diff,
    color_superdiff,
    color_acmin,
    color_turb,
    MidpointNormalize,
)
from utils import process_eigen_grad, row_norm, check_commutation
from timer import timeit_debug, timeit_info
from autocorrelation import (
    zcf_gen,
    zcf_sum,
    acf_gen,
    acf_sum,
    constraints_stochmat,
    bounds_statdist,
)


class Propagator(object):
    """
    FUNCTION: Propagates, enough said.
    INPUTS: GEN = generator.
    """

    @timeit_info
    def __init__(
        self,
        GEN,
        nu=None,
        sigma=1.0,
        tau=1.0,
        alpha=1.0,
        beta=1.0,
        spec_noise=0.0,
        power_spec=None,
        strict=False,
        label=None,
    ):
        """
        FUNCTION: Processes multiple Q-generators.
        INPUTS: GEN         = generator instance
                nu          = if not None, sigma scaling as a function of n_state (over-rides sigma)
                sigma       = spatial constant (diffusion speed/scale is correlated with sigma^2)
                tau         = tempo parameter (diffusion speed/scale is inversely correlated with tau)
                alpha       = stability parameter
                beta        = softmax normalization temperature
                spec_noise  = add zero-mean noise to power spectrum with variance spec_noise
                power_spec    = use specific power spectrum
                strict      = strict checks on propagator kernel
                label       = propagator description (e.g. diffusion/superdiffusion/turbulence), if None determined by alpha
        NOTES: inverse tau  = amount of time per circuit iteration
               sigma and tau are not exactly inversely related in anomalous regimes
        """
        self.GEN = GEN
        self.nu = nu
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.spec_noise = float(spec_noise)
        self.strict = strict
        self.states = np.array(list(range(self.n_state)))
        if label is None:
            if alpha == 1:
                self.label = "diffusion"
            elif alpha < 1:
                self.label = "superdiffusion"
            elif alpha > 1:
                self.label = "turbulence"
        else:
            self.label = label

        if nu is not None:
            assert 0 < self.nu <= 1, "PROPAGATOR: nu out of range"
            self.sigma = np.sqrt(self.nu * self.n_state / 2.0)

        assert 0 < self.sigma, "PROPAGATOR: sigma out of range"
        assert 0 < self.tau, "PROPAGATOR: tau out of range"
        assert 0 < self.alpha, "PROPAGATOR: alpha out of range"
        assert 0 < self.beta <= 1, "PROPAGATOR: beta out of range"

        # secondary parameters
        self.secondary_params()

        # construct propagator
        self.compute_kernels(power_spec=power_spec, strict=strict)

        # scale at which process msd saturates (i.e. expected squared displacement as t->infty)
        self.msd_saturation = (
            self.msd_saturation_time()
        )  # self.n_state as a proxy for the square root of the total size of the environment

        # set color scheme
        if self.alpha == 1:
            self.color = color_diff
        elif self.alpha < 1:
            self.color = color_superdiff
        elif self.alpha > 1:
            self.color = color_turb
        self.set_target_axis()

    @property
    def n_state(self):
        return self.GEN.n_state

    @property
    def n_dim(self):
        return self.ENV.n_dim

    @property
    def ENV(self):
        return self.GEN.ENV

    @property
    def world_array(self):
        return self.GEN.ENV.world_array

    @property
    def U(self):
        return self.GEN.EVEC_fwd

    @property
    def Uinv(self):
        return self.GEN.EVECinv_fwd

    @property
    def L(self):
        return self.GEN.evals_fwd

    @property
    def U_bwd(self):
        return self.GEN.EVEC_bwd

    @property
    def Uinv_bwd(self):
        return self.GEN.EVECinv_bwd

    @property
    def L_bwd(self):
        return self.GEN.evals_bwd

    def set_new_spec_mod(self, tau=None, alpha=None, power_spec=None):
        """ set new power spectrum or diffusive regime """
        if tau is not None:
            self.tau = float(tau)
        if alpha is not None:
            self.alpha = float(alpha)
        self.secondary_params()
        self.compute_kernels(power_spec=power_spec, strict=self.strict)

    def secondary_params(self):
        """ computes secondary parameters from primary sigma/alpha/tau/L """
        # factor of 2 since "gaussian" parametrized as alpha=1 in discrete case
        self.sigma_alpha = self.sigma ** (2 * self.alpha)
        # alpha-modulated diffusion constant
        self.K_alpha = self.sigma_alpha / self.tau
        # scale eigenvalues by stability parameter alpha
        self.L_alpha = np.abs(self.L) ** (self.alpha)

    @timeit_debug
    def spectral_density(self, t=1, k=None):
        """computes the spectral density as a function of time displacement t and spectral component k
        depends on alpha/beta/sigma and generator eigenvalues"""
        assert hasattr(self, "L_alpha"), "L_alpha unavailable?"
        # dilation in frequency space
        x = self.sigma_alpha * self.L_alpha * t / self.tau
        d = np.exp(-x)
        if k is None:
            return d
        else:
            return d[k]

    @timeit_debug
    def weight_spec_comps(self, eigen_grad=None):
        """
        FUNCTION: Weights propagator kernels.
        INPUTS: eigen_grad = desired gradient on eigen-decomposition
                None implies all weights equal to 1
        NOTES:
        """
        self.eigen_grad = eigen_grad
        self.n_kernels = self.n_state
        if eigen_grad is None:
            self.spec_comp_weights = np.ones((self.n_kernels,))
        else:
            self.spec_comp_weights = process_eigen_grad(self.eigen_grad, self.n_state)

    @timeit_debug
    def set_power_spec(self, power_spec=None):
        if power_spec is None:
            self.power_spec = self.spectral_density(t=1.0)
        else:
            assert power_spec.size == self.n_state, "power spectrum wrong shape"
            self.power_spec = power_spec
        if self.spec_noise != 0.0:
            self.power_spec_noisefree = self.power_spec.copy()
            self.power_spec *= np.random.normal(
                loc=1, scale=self.spec_noise, size=self.power_spec.size
            )
            self.power_spec = self.power_spec.clip(
                self.power_spec_noisefree.min(), None
            )

    @timeit_debug
    # @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def compute_kernels(
        self,
        power_spec=None,
        suppress_imag=True,
        strict=False,
        atol=1.0e-2,
        rtol=1.0e-2,
    ):
        """
        FUNCTION: Computes propagator kernels.
        INPUTS: power_spec = power spectrum, None implies computed from alpha/tau etc
                suppress_imag = suppress any imaginary components
                strict = True implies extra checks on propagator structure
        NOTES: Depends on sigma/tau/alpha
        """
        # self.L, self.U, self.Uinv set as properties
        # set propagator kernel weights
        self.weight_spec_comps()
        self.set_power_spec(power_spec=power_spec)
        self.etD = (
            np.eye(self.n_state) * self.power_spec
        )  # spectral power as a diagonal matrix
        # re-weighting
        self.wetD = self.spec_comp_weights * self.etD
        # map onto basis set in frequency space "decayed" in time
        self.spec_basis = np.matmul(self.U, self.wetD)
        # propagator (i.e. map to basis set in state-space future time)
        self.etO = np.matmul(self.spec_basis, self.Uinv)
        if suppress_imag:
            if not np.all(np.isreal(self.etO)):
                etO_complex = deepcopy(self.etO)
                print("PROPAGATOR: squashing imaginary components.")
                # self.etO = self.etO.real
                self.etO = np.abs(self.etO)  # seems to work better
                if strict:
                    assert np.allclose(
                        self.etO, etO_complex
                    ), "PROPAGATOR: propagator kernel is complex."
        if strict:
            assert np.allclose(self.etO.min(), 0, atol=atol, rtol=rtol), (
                "PROPAGATOR: propagator taking values %.2f significantly <0"
                % self.etO.min()
            )
            assert (self.etO <= 1).all(), "PROPAGATOR: propagator kernel values > 1."
            assert np.allclose(
                self.etO.sum(1), 1, atol=atol, rtol=rtol
            ), "PROPAGATOR: probability density not conserved."
        self.activation_matrix()

    @timeit_info
    def min_zero_cf(self, lags=[1], rho_init="stationary", maxiter=1000):
        """
        FUNCTION: sets spectrum to minimize zero-time correlation at lags
        INPUTS: lags = list of time offsets, ACF sum over lags is minimized
                rho_init = ACF initialized from this distribution, 'start', 'stationary', 'uniform', int (state), array
                maxiter     = maximum # iterations during optimization
        """
        if type(rho_init) is str:
            if rho_init == "stationary":
                rho0 = self.GEN.stationary_dist()  # stationary density
            elif rho_init == "uniform":
                rho0 = np.ones((self.n_state,)) / float(self.n_state)
            elif rho_init == "start":
                rho0 = np.zeros(self.n_state)
                rho0[self.ENV.start] = 1.0
            else:
                raise ValueError(
                    "unknown setting for initial distribution in acf calculation"
                )
        elif type(rho_init) is int:
            rho0 = np.zeros(self.n_state)
            rho0[rho_init] = 1.0
        elif rho_init.size == self.n_state:
            rho0 = rho_init
        else:
            raise ValueError(
                "unknown setting for initial distribution in acf calculation"
            )

        x0 = self.power_spec  # initialize at currently set spectrum
        W = self.GEN.spectral_matrix()
        fun = partial(zcf_sum, W=W, deltaT=lags, rho=rho0)
        options = {"disp": True, "maxiter": maxiter}

        # 1. evolution matrix is a stochastic matrix
        # 2. stationary distribution is preserved
        # local optimization
        # lc1_stochmat, lc2_stochmat = constraints_stochmat(W)
        # bounds = bounds_statdist(self.n_state)
        # opt = minimize(fun, x0, method=None, bounds=(bounds), constraints=(lc1_stochmat, lc2_stochmat), tol=None, callback=None, options=options)

        # 1. evolution matrix is a stochastic matrix
        lc1_stochmat, lc2_stochmat = constraints_stochmat(W)
        opt = minimize(
            fun,
            x0,
            method=None,
            constraints=(lc1_stochmat, lc2_stochmat),
            tol=None,
            callback=None,
            options=options,
        )

        # 2. stationary distribution is preserved
        # opt = minimize(fun, x0, method=None, bounds=(bounds), tol=None, callback=None, options=options)

        # no constraints
        # opt = minimize(fun, x0, method=None, tol=None, callback=None, options=options)

        print(opt)
        s_opt = opt.x
        self.compute_kernels(power_spec=s_opt)
        self.color = color_acmin

    @timeit_info
    def min_auto_cf(self, T=2, lags=[1, 2], rho_init="stationary", maxiter=1000):
        """
        FUNCTION: sets spectrum to minimize autocorrelation at lags summed over times
        INPUTS: T           = maximum sampled timesteps to consider ACF
                lags        = list of time offsets, ACF sum over lags is minimized
                rho_init    = ACF initialized from this distribution, 'start', 'stationary', int (state), array
                maxiter     = maximum # iterations during optimization
        """
        # FIXME inequality constraints violated using scipy.minimize
        # https://scikit-optimize.github.io - forest_minimize broken
        # TODO try hyperopt, pyomo, gurobi?

        n_k = self.n_state
        if type(rho_init) is str:
            if rho_init == "stationary":
                rho0 = self.GEN.stationary_dist()  # stationary density
            elif rho_init == "start":
                rho0 = np.zeros(self.n_state)
                rho0[self.ENV.start] = 1.0
            else:
                raise ValueError(
                    "unknown setting for initial distribution in acf calculation"
                )
        elif type(rho_init) is int:
            rho0 = np.zeros(self.n_state)
            rho0[rho_init] = 1.0
        elif rho_init.size == self.n_state:
            rho0 = rho_init
        else:
            raise ValueError(
                "unknown setting for initial distribution in acf calculation"
            )

        x0 = self.power_spec  # initialize at currently set spectrum
        W = self.GEN.spectral_matrix()
        fun = partial(acf_sum, W=W, T=T, deltaT=lags, rho=rho0)
        options = {"disp": True, "maxiter": maxiter}

        # 1. evolution matrix is a stochastic matrix
        # 2. stationary distribution is preserved
        # local optimization
        # lc1_stochmat, lc2_stochmat = constraints_stochmat(W)
        # bounds = bounds_statdist(n_k)
        # opt = minimize(fun, x0, method=None, bounds=(bounds), constraints=(lc1_stochmat, lc2_stochmat), tol=None, callback=None, options=options)

        # 1. evolution matrix is a stochastic matrix
        lc1_stochmat, lc2_stochmat = constraints_stochmat(W)
        opt = minimize(
            fun,
            x0,
            method=None,
            constraints=(lc1_stochmat, lc2_stochmat),
            tol=None,
            callback=None,
            options=options,
        )

        # 2. stationary distribution is preserved
        # opt = minimize(fun, x0, method=None, bounds=(bounds), tol=None, callback=None, options=options)

        # no constraints
        # opt = minimize(fun, x0, method=None, tol=None, callback=None, options=options)

        print(opt)
        s_opt = opt.x
        self.compute_kernels(power_spec=s_opt)
        self.color = color_acmin

    def predict_acf(self, lags=[1], subtract_stat_dist=False, rho_init="stationary"):
        """
        FUNCTION: predicts autocorrelation function from propagator design
        INPUTS: lags = list of time displacements over which to compute ACF
                subtract_stat_dist = subtract stationary distribution from output
                rho_init = ACF initialized from this distribution, 'start', 'stationary', int (state)
        """
        if type(rho_init) is str:
            if rho_init == "stationary":
                rho0 = self.GEN.stationary_dist()  # stationary density
            else:
                raise ValueError(
                    "unknown setting for initial distribution in acf calculation"
                )
        elif type(rho_init) is int:
            rho0 = np.zeros(self.n_state)
            rho0[rho_init] = 1.0
        elif rho_init.size == self.n_state:
            rho0 = rho_init
        else:
            raise ValueError(
                "unknown setting for initial distribution in acf calculation"
            )
        W = self.GEN.spectral_matrix()
        Cpred = zcf_gen(self.power_spec, W, lags, rho0)

        if subtract_stat_dist:
            rho_inf = self.GEN.stationary_dist()
            n_t = Cpred.size
            return Cpred.sum() - rho_inf.mean() * n_t
        else:
            return Cpred

    def activation_matrix(self, thresh=0.1):
        """ self.AMT = thresholded activation matrix at self.etO>thresh"""
        assert 0 < thresh < 1, "thresh parameter out of bounds"
        self.thresh = thresh
        self.AMT = self.etO >= thresh

    def shift_norm_prop(self):
        """ shifts self.etO into non-negative range and row-normalizes. """
        self.etO += repmat(self.etO.min(1), self.n_state, 1)
        self.etO = row_norm(self.etO)

    def set_target_axis(self, ax=None):
        """
        FUNCTION: Sets whether a fig/axis is to be used for plotting output.
        """
        if ax is None:
            self.fig = None
            self.ax = None
            self.no_target_axis = True
        else:
            self.fig = plt.gcf()
            self.ax = ax
            self.no_target_axis = False

    def fig_axis(self):
        """
        FUNCTION: Sets figure and/or axis to plot to.
        """
        if self.no_target_axis:
            fig, ax = plt.subplots(figsize=(10, 10))
            self.fig = fig
            self.ax = ax
        else:
            plt.axes(self.ax)

    def plot_grid_gains(self):
        """
        FUNCTION: Plots the gains on the grid cells (i.e. the weighted/time-decayed evals).
        """
        self.fig, self.axs = plt.subplots(
            nrows=3, ncols=1, sharex=False, figsize=(5, 12)
        )

        # plot weighted exponential decays per frequency across time (one panel)
        self.axs[0].plot(sorted(self.GEN.evals), np.diag(self.wetD), "-o", color="k")
        self.axs[0].set_xlabel("eigenvalue $ \lambda_k$")
        self.axs[0].set_ylabel("Gain on grid cell output $ w_ke^{t \lambda_k}$")
        self.axs[0].invert_xaxis()
        sb.despine(ax=self.axs[0], top=True, right=True, left=False, bottom=False)

        # plot frequency weights per spectral component (one panel)
        self.axs[1].plot(self.spec_comp_weights, "-o", color="k")
        self.axs[1].set_xlabel("spectral component")
        self.axs[1].set_ylabel("spectral weight")
        sb.despine(ax=self.axs[1], top=True, right=True, left=False, bottom=False)

        # plot decays per spectral component across time (one panel)
        self.axs[2].plot(np.diag(self.etD), "-o", color="k")
        self.axs[2].set_xlabel("spectral component $ k$")
        self.axs[2].set_ylabel("temporal decay factor $ e^{t \lambda_k}$")
        sb.despine(ax=self.axs[2], top=True, right=True, left=False, bottom=False)

        # plt.suptitle('Grid cell gains (controls time decay)', fontsize=suptitle_fontsize)
        self.fig.tight_layout()

    def plot_spectral_scaling(self):
        """
        FUNCTION: Plots the space/time scaling across spectral components.
        NOTES: essentially replaces plot_grid_gains
        """
        self.fig, self.axs = plt.subplots(
            nrows=3, ncols=1, sharex=False, figsize=(5, 12)
        )

        # plot weighted exponential decays per frequency across time (one panel)
        self.axs[0].plot(sorted(self.GEN.evals), np.diag(self.wetD), "-o", color="k")
        self.axs[0].set_xlabel("eigenvalue $ \lambda_k$")
        self.axs[0].set_ylabel("Gain on grid cell output $ w_ke^{t \lambda_k}$")
        self.axs[0].invert_xaxis()
        sb.despine(ax=self.axs[0], top=True, right=True, left=False, bottom=False)

        # plot frequency weights per spectral component (one panel)
        self.axs[1].plot(self.spec_comp_weights, "-o", color="k")
        self.axs[1].set_xlabel("spectral dimension $ k$")
        self.axs[1].set_ylabel("spectral weight")
        self.axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        sb.despine(ax=self.axs[1], top=True, right=True, left=False, bottom=False)

        # plot decays per spectral component across time (one panel)
        self.axs[2].plot(np.diag(self.etD), "-o", color="k")
        self.axs[2].set_xlabel("spectral dimension $ k$")
        self.axs[2].set_ylabel("temporal decay factor $ e^{t \lambda_k}$")
        self.axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        sb.despine(ax=self.axs[2], top=True, right=True, left=False, bottom=False)

        # plt.suptitle('Grid cell gains (controls time decay)', fontsize=suptitle_fontsize)
        self.fig.tight_layout()

    def plot_spectral_basis(self, n=9, wrap_col=3, time_decay=True):
        """
        FUNCTION: plots spectral basis vectors
        INPUTS: n = number of spectral basis vectors
                wrap_col = max number of columns
                time_decay = plot spectral basis set decayed in time or not
        """
        if time_decay:
            FK = (
                self.spec_basis.real
            )  # FIXME separately record real/imag frequency kernels as dilations/rotations
        else:
            FK = self.U.real
        vmin = FK[:, :n].flatten().min()
        vmax = FK[:, :n].flatten().max()
        nrow, ncol = grid_wrap_nrow(total=n, wrap_col=wrap_col)
        self.fig, self.axs = plt.subplots(
            nrows=nrow,
            ncols=ncol,
            gridspec_kw=gridspec_kw,
            sharex=False,
            sharey=False,
            figsize=figsize,
        )
        self.axs = self.axs.reshape(-1)

        # plot kernels
        if np.iscomplex(self.spec_basis).any():
            print(
                "propagators.plot_spectral_basis: eigenvectors are complex, only real components plotted."
            )
        for ix in range(n - 1):
            ax = self.axs[ix]
            ax = self.ENV.plot_state_func(
                state_vals=self.spec_basis[:, ix].real,
                ax=ax,
                cmap=cmap_grid_code,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
            )
            ax.set_title("spectral basis vector %i" % ix)
        ix = n - 1
        ax = self.axs[ix]
        ax = self.ENV.plot_state_func(
            state_vals=self.spec_basis[:, ix].real,
            ax=ax,
            cmap=cmap_grid_code,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
        )
        ax.set_title("spectral basis vector %i" % ix)
        # plt.suptitle('spectral basis vectors',y=1.02, fontsize=suptitle_fontsize)
        self.fig.tight_layout()

    def plot_prop_kernels(
        self,
        n=6,
        first_state=0,
        wrap_col=6,
        autoprop_off=False,
        cmap=plt.cm.RdBu_r,
        midpoint_norm=False,
        cbar=False,
        vmin=None,
        vmax=None,
    ):
        """
        FUNCTION: plots "propagator kernels" (i.e. local propagators).
        INPUTS: n = number of kernels
                first_state = number of kernel to start with
                wrap_col = max number of columns
                autoprop_off = suppress propagation to current position
                cbar = True includes colorbar
                vmin/vmax = sets limits of color scale
        NOTES: propagator kernels are state-dependent in inhomogeneous domains
        """
        etO = self.etO.copy()
        if autoprop_off:
            np.fill_diagonal(etO, 0.0)
        if vmin is None:
            vmin = etO[first_state : first_state + n, :].real.flatten().min()
        if vmax is None:
            vmax = etO[first_state : first_state + n, :].real.flatten().max()

        if self.no_target_axis:
            nrow, ncol = grid_wrap_nrow(total=n, wrap_col=wrap_col)
            self.fig, self.axs = plt.subplots(
                nrows=nrow,
                ncols=ncol,
                gridspec_kw=gridspec_kw,
                sharex=False,
                sharey=True,
                figsize=(page_width, nrow * row_height),
            )
            if n == 1:
                self.axs = [self.axs]
            else:
                self.axs = self.axs.reshape(-1)
        else:
            self.axs = [self.ax]

        if midpoint_norm:
            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
        else:
            norm = None

        for ix in range(n):
            kix = first_state + ix
            ax = self.axs[ix]
            plt.sca(ax)
            ax = self.ENV.plot_state_func(
                state_vals=etO[kix, :].real,
                ax=ax,
                cbar=cbar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                mask_color="white",
                norm=norm,
                arrows=False,
            )
            # ax.set_title("state %i" % kix)
        if self.no_target_axis:
            # plt.suptitle('Propagator kernels',y=1.02, fontsize=suptitle_fontsize)
            self.fig.tight_layout()

    def compute_power_spectrum(self, alpha_base=1.0, tau_base=1.0):
        """
        FUNCTION: computes (relative) power spectrum.
        OUTPUT: sets df_spec.
        """
        self.alpha_base = alpha_base
        self.tau_base = tau_base
        alphas = [self.alpha, alpha_base]
        taus = [self.tau, tau_base]

        # power spectrum
        n_x = 100
        n_state = self.n_state
        evals = self.L
        x_state = np.linspace(0, 1, n_state)
        x = np.linspace(0, 1, n_x)
        comps = range(1, n_state + 1)
        # generate pandas dataaset
        # iterables_eval = [alphas, taus, comps]
        # ix_eval = pd.MultiIndex.from_product(iterables_eval, names=['alpha','tau','comp'])
        iterables_spec = [[self.label, "base"], comps]
        ix_spec = pd.MultiIndex.from_product(iterables_spec, names=["regime", "x"])
        df_spec = pd.DataFrame(
            index=ix_spec,
            columns=["eval", "evec_no", "alpha", "tau", "gain", "log_gain"],
            dtype="float",
        )

        # propagator power spectrum "gain"
        df_spec.loc[pd.IndexSlice[self.label, :], "eval"] = evals
        df_spec.loc[pd.IndexSlice[self.label, :], "evec_no"] = comps
        df_spec.loc[pd.IndexSlice[self.label, :], "alpha"] = self.alpha
        df_spec.loc[pd.IndexSlice[self.label, :], "tau"] = self.tau
        df_spec.loc[pd.IndexSlice[self.label, :], "gain"] = np.diag(self.wetD)
        df_spec.loc[pd.IndexSlice[self.label, :], "log_gain"] = np.log(
            np.diag(self.wetD)
        )
        # baseline power spectrum "gain"
        PROP_base = Propagator(
            GEN=self.GEN,
            sigma=self.sigma,
            tau=tau_base,
            alpha=alpha_base,
            beta=self.beta,
            label="base",
        )
        df_spec.loc[pd.IndexSlice["base", :], "eval"] = evals
        df_spec.loc[pd.IndexSlice["base", :], "evec_no"] = comps
        df_spec.loc[pd.IndexSlice["base", :], "alpha"] = self.alpha_base
        df_spec.loc[pd.IndexSlice["base", :], "tau"] = self.tau_base
        df_spec.loc[pd.IndexSlice["base", :], "gain"] = np.diag(PROP_base.wetD)
        df_spec.loc[pd.IndexSlice["base", :], "log_gain"] = np.log(
            np.diag(PROP_base.wetD)
        )

        # remove stationary eigenvector
        df_spec = df_spec[df_spec.evec_no != self.n_state + 1]

        # flatten dataframes
        df_spec = pd.DataFrame(df_spec.to_records())
        # df_spec_base = df_spec[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_base)]
        df_spec_base = df_spec[df_spec.regime == "base"]

        # gain relative to alpha=alpha_base/tau=tau_base baseline
        df_spec_base = df_spec_base.loc[
            (df_spec_base.alpha == alpha_base) & (df_spec_base.tau == tau_base)
        ]
        df_spec_rel = df_spec.copy()
        df_spec_rel["log_gain"] = df_spec_rel.groupby(["regime"])["log_gain"].apply(
            lambda x: x - df_spec_base.log_gain.values
        )
        df_spec_rel["gain"] = df_spec_rel.groupby(["regime"])["gain"].apply(
            lambda x: x / df_spec_base.gain.values
        )

        # gain relative to total absolute gain
        df_spec_rel["log_gain"] = df_spec_rel.groupby(["regime"])["log_gain"].apply(
            lambda x: x - x.sum()
        )
        df_spec_rel["gain"] = df_spec_rel.groupby(["regime"])["gain"].apply(
            lambda x: x / x.sum()
        )

        self.df_spec = df_spec
        self.df_spec_rel = df_spec_rel

    def plot_power_spectrum(
        self, x_index="eval", loglog=True, target_ax=None, **kwargs
    ):
        """
        FUNCTION: plots power spectrum.
        INPUTS: target_ax = axis to plot to.
                x_index = eval or evec_no
                loglog = plot in log space
        """
        if not hasattr(self, "df_spec"):
            self.compute_power_spectrum()

        if target_ax is None:
            sb.set_style("ticks")
            fig, target_ax = plt.subplots(
                nrows=1, ncols=1, figsize=(page_width, row_height)
            )
            sb.despine(fig, top=True, right=True)

        ax = sb.lineplot(
            data=self.df_spec,
            x=x_index,
            y="gain",
            hue="regime",
            ax=target_ax,
            marker="o",
            **kwargs
        )
        ax.set_xlim([self.df_spec[x_index].min(), self.df_spec[x_index].max()])
        if x_index == "eval":
            # ax.set_xlabel('eigenvalue $ \lambda$')
            ax.set_xlabel(r"spectral wavelength $\lambda$")
            ax.set_xticks([ax.get_xticks()[0] + 0.5, ax.get_xticks()[-1] - 0.3])
            ax.tick_params(axis="x", length=0)
            ax.set_xticklabels(["short", "long"])
            ax.set_ylabel(r"$d_{\tau,\alpha}(\lambda)$")
        else:
            ax.set_xlabel(r"spectral wavelength $k$")
            ax.set_ylabel(r"$d_{\tau,\alpha}(k)$")
            ax.tick_params(axis="x", length=0)
            ax.set_xticklabels(["short", "long"])
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        sb.despine(ax=ax, right=True, top=True)
        # remove legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        ax.set_title("power spectrum")

    def plot_relative_power_spectrum(
        self,
        x_index="eval",
        xlog=False,
        ylog=False,
        target_ax=None,
        color=None,
        plot_base=True,
        **kwargs
    ):
        """
        FUNCTION: plots relative power spectrum.
        INPUTS: target_ax = axis to plot to
                color = None, inferred from alpha
                x_index = eval or evec_no
                xlog/ylog = plot in log space
                plot_base = plot relative power spectrum for baseline tau=1,alpha=1
        """
        if not hasattr(self, "df_spec_rel"):
            self.compute_power_spectrum()

        if target_ax is None:
            sb.set_style("ticks")
            fig, target_ax = plt.subplots(
                nrows=1, ncols=1, figsize=(page_width, row_height)
            )
            sb.despine(fig, top=True, right=True)

        if color is None:
            color = self.color

        if plot_base:
            ax = sb.lineplot(
                data=self.df_spec_rel[self.df_spec_rel.regime == "base"],
                x=x_index,
                y="gain",
                ax=target_ax,
                marker="o",
                color="black",
                markeredgewidth=0.0,
                markerfacecolor="black",
                markeredgecolor="black",
                clip_on=False,
                zorder=100,
                label="baseline",
                **kwargs
            )

        ax = sb.lineplot(
            data=self.df_spec_rel[self.df_spec_rel.regime == self.label],
            x=x_index,
            y="gain",
            ax=target_ax,
            marker="o",
            color=color,
            markeredgewidth=0.0,
            markerfacecolor=color,
            markeredgecolor=color,
            clip_on=False,
            zorder=100,
            label=self.label,
            **kwargs
        )

        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")

        if x_index == "eval":
            ax.set_xlabel(r"eigenvalue $ \lambda$")
            # ax.set_xlabel(r'spectral wavelength $\lambda$')
            # ax.set_ylabel(r"$r_{\tau,\alpha}(\lambda)$   [normalized]")
            ax.set_ylabel(r"$s_{\tau,\alpha}(\lambda)/s_{1,1}(\lambda)$   [normalized]")
        else:
            ax.set_xlabel(r"spectral component no. $k$")
            ax.set_xticks([self.n_state, 1])
            ax.set_xticklabels(["short", "long"])
            ax.tick_params(axis="x", length=0)
            # ax.set_ylabel(r"$r_{\tau,\alpha}(k)$   [normalized]")
            ax.set_ylabel(r"$s_{\tau,\alpha}(k)/s_{1,1}(k)$   [normalized]")
        # remove legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        sb.despine(ax=ax, right=True, top=True)
        ax.set_title("relative power spectrum")

    def plot_prop_kernels_matrix(
        self, off_diagonal=True, offdiag_cbar=True, interpolation="spline36",
         state_ticks=[1,8,17,26],
         state_ticklabels=['X', 'G1', 'G2', 'G3']
    ):
        """
        FUNCTION: plots propagator in matrix form
        INPUTS: off_diagonal = True zero out diagonal (self-activation)
                off_diag_cbar = set vmin/vmax of colorbar based on offdiagonals
                interpolation = interploation in imshow
        """
        if self.no_target_axis:
            self.fig = plt.figure(figsize=figsize)
            self.ax = plt.gca()

        if offdiag_cbar:
            X = self.etO.copy()
            np.fill_diagonal(X, 0)
            vmin = X.min()
            vmax = X.max()
        else:
            vmin = None
            vmax = None
        if off_diagonal:
            X = self.etO.copy()
            np.fill_diagonal(X, 0)
        else:
            X = self.etO.copy()
        # plot
        improp = self.ax.imshow(
            X,
            origin="upper",
            interpolation=interpolation,
            cmap=cmap_activation_prob,
            vmin=vmin,
            vmax=vmax
        )
        self.ax.set_ylabel("current position")
        self.ax.set_xlabel("future position")
        # plt.title('propagator kernels', pad=20)
        self.ax.set_title("propagator", pad=10)
        self.fig.colorbar(mappable=improp, shrink=1., ax=self.ax)
        # self.ax.xaxis.set_label_position('top')
        # self.ax.xaxis.tick_top()
        self.ax.grid(color="gray", linestyle="-", linewidth=0.5)
        self.ax.plot(list(reversed(self.ax.get_xlim())), self.ax.get_ylim(), ls="-", c="white", linewidth=0.5)
        if not state_ticks is None:
            # goal-specific labeling
            self.ax.set_xticks(state_ticks)
            self.ax.set_yticks(state_ticks)
            self.ax.set_xticklabels(state_ticklabels)
            self.ax.set_yticklabels(state_ticklabels)

    def plot_activation_matrix(self, thresh=0.1):
        """plots propagator in matrix form"""
        self.activation_matrix(thresh=thresh)
        sb.set(style="dark", font_scale=font_scale)
        self.fig = plt.figure(figsize=figsize)
        plt.imshow(self.AMT, origin="upper")
        plt.ylabel("current position")
        plt.xlabel("future position")
        plt.title("activation probability > %.1f" % thresh, pad=20)
        # plt.gca().xaxis.set_label_position('top')
        # plt.gca().xaxis.tick_top()
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_limit_density(self, plot_env=False, vmin=0, vmax=None):
        """
        FUNCTION: plots the state density as t->infty.
        INPUTS: plot_env = True/False, whether or not to plot on top of plot_environment
                gridworlds plot on top of environment automatically
        NOTES: corresponds to square of zeroth eigenvector
        """
        self.limit_density = (self.U[:, 0]) ** 2
        assert np.allclose(
            self.limit_density.sum(), 1
        ), "Stationary density does not sum to 1"
        self.fig_axis()
        if plot_env:
            self.ENV.plot_environment(ax=self.ax)
        self.ENV.plot_state_func(
            state_vals=self.limit_density,
            vmin=vmin,
            vmax=vmax,
            ax=self.ax,
            cbar=True,
            cbar_label="Stationary probability",
            arrows=False,
        )
        self.ax.set_title("Stationary state density in limit of infinite time")

    def msd(self, t_steps=1.0, K=1.0, logspace=False, base="e"):
        """
        analytically computes mean-squared (position) displacement as a function of time displacement
        msd(t) = alpha\inv \log{\tau\inv t} + \log{K}
        INPUTS: t_steps = time displacements to compute
                K != None over-rides K:=self.sigma**2 diffusion constant (which sometimes causes flow problems in kernel calculations)
                logspace = True returns result in log-space
                base = base of log calculation, exp by default
        """
        if base == 10:
            log = np.log10
        elif base == 2:
            log = np.log2
        elif base == "e":
            log = np.log
        if K == None:
            K = self.sigma_alpha
        if logspace:
            return (1 / self.alpha) * log(t_steps / self.tau) + log(K)
        else:
            return K * (t_steps / self.tau) ** (1 / self.alpha)

    def mean_displacement(self, t_steps=1.0, K=1.0, logspace=False, base="e"):
        """
        analytically computes mean (position) distance as a function of time displacement
        mean_displacement(t) = (2 alpha) \inv \log{\tau\inv t} + 0.5\log{K}
        INPUTS: t_steps = time displacements to compute
                K != None over-rides K:=self.sigma**2 diffusion constant (which sometimes causes flow problems in kernel calculations)
                logspace = True returns result in log-space
                base = base of log calculation, exp by default
        """
        if base == 10:
            log = np.log10
        elif base == 2:
            log = np.log2
        elif base == "e":
            log = np.log
        if K == None:
            K = self.sigma_alpha
        if logspace:
            return (1 / 2 * self.alpha) * log(t_steps / self.tau) + 0.5 * log(K)
        else:
            return np.sqrt(K * (t_steps / self.tau) ** (1 / self.alpha))

    def msd_saturation_time(self, state_size=1.0, sigma=None):
        """self.n_state*state_size as a proxy for the square root of the total size of the environment"""
        if sigma is None:
            sigma = self.sigma
        return (self.tau / (sigma ** 2)) * (state_size * self.n_state) / 6.0

    def autocorrelation_time(self, t=1.0, state=None):
        """
        FUNCTION: returns the discrete autocorrelation function for all states (state=None) or a specified state at a given time displacement t
        """
        d = self.spectral_density(t=t, k=None)
        # C = d@self.U.T # d indexed by spec component, U indexed state x spec component (low to high wavelength), C indexed by state
        C = (
            d @ self.Uinv
        )  # inverse transform, d indexed by spec component, U indexed state x spec component (low to high wavelength), C indexed by state
        # import pdb; pdb.set_trace()
        if state == None:
            return C
        else:
            return C[state]

    def autocorrelation(self, state, t_max=10.0, n_val=20):
        """
        FUNCTION: autocorrelation as a function of time for a given state
        """
        ts = np.linspace(0, t_max, n_val)
        C = np.zeros((n_val,))
        for i, t in enumerate(ts):
            C[i] = self.autocorrelation_time(t=t, state=state)
        return C

    def plot_autocorr(self, state, t_max=10.0, n_val=20):
        """
        FUNCTION: plot autocorrelation for state in time interval [0,t_max] based on n_val samples
        """
        ts = np.linspace(0, t_max, n_val)
        C = self.autocorrelation(state=state, t_max=t_max, n_val=n_val)
        self.fig_axis()
        self.ax.plot(ts, C, "-o", color="black")
        self.ax.set_xlabel(r"$\Delta t$")
        self.ax.set_ylabel(r"$C(x,\Delta t)$")
        self.ax.axhline(0)
        self.ax.set_title("spatial autocorrelation")
        sb.despine(top=True, right=True)


def copy_propagator(PROP):
    from copy import deepcopy

    PROP.fig = None
    PROP.ax = None
    return deepcopy(PROP)


def compose_comm_propagators(PROPs=[]):
    """
    FUNCTION: composes commuting propagators
    INPUT: PROPs = propagators to compose
    NOTES: defines new propagator corresponding to composition of PROPs which inherits member instances from PROPs[0]
    """
    from utils import check_commutation
    from itertools import product
    from functools import reduce

    for pair in product(PROPs, PROPs):
        assert check_commutation(pair[0].etO, pair[1].etO), "propagators do not commute"
    etOs = [PROP.etO for PROP in PROPs]
    PROPc = copy_propagator(PROPs[0])
    PROPc.etO = reduce(np.dot, etOs)
    return PROPc
