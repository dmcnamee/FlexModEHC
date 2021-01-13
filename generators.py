# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA
import seaborn as sb
import matplotlib.pyplot as plt
import visualization as vis

from copy import deepcopy

import os

import config
from visualization import (
    grid_wrap_nrow,
    gridspec_kw,
    figsize,
    cmap_grid_code,
    page_width,
    row_height,
)
from utils import row_norm, eigen_decomp, is_symmetric, symnorm_graph_laplacian, is_normal, is_unitary
from timer import timeit_debug, timeit_info


class Generator(object):
    @timeit_info
    def __init__(
        self,
        ENV=None,
        Q=None,
        T=None,
        W=None,
        jump_rate=15.0,
        forward=True,
        symmetrize=False,
    ):
        """
        FUNCTION: Constructs eigensystem for infinitesimal generator corresponding to underlying state-space.
        INPUTS: ENV = environment with accessibility matrix ENV.A
                Q = explicit generator
                T = stochastic matrix
                W = weight matrix
                jump_rate = rate at which particle jumps from current state (higher = more likely to jump)
                            (only applies if generator is being constructed from DTMC e.g. T/A)
                forward = True, forward in time => right-multiplying matrix
                symmetrize = symmetrize generator
        NOTES: State-space can be specified in terms of an environment ENV, generator Q, stochastic matrix T, or weight matrix W.
        Primacy given to environment variable ENV.
        """
        if ENV is not None:
            if hasattr(ENV, "T"):
                self.T = ENV.T
                Q = stochmat2generator(T=self.T, jump_rate=jump_rate)
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from environment transition matrix with jump_rate %.2f"
                    % jump_rate
                )
            else:
                Q = adjmat2generator(ENV.A_adj, jump_rate=jump_rate)
                self.T = generator2stochmat(Q)
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from environment adjacency matrix with jump_rate %.2f"
                    % jump_rate
                )
            self.ENV = ENV
        elif Q is None:
            if T is not None:
                Q = stochmat2generator(T=T, jump_rate=jump_rate)
                self.T = T
                self.W = generator2weightmat(Q)
                print(
                    "GENERATOR: generator constructed from arbitrary transition matrix with jump_rate %.2f"
                    % jump_rate
                )
            else:
                Q = weightmat2generator(W=W, normsym=symmetrize)
                self.W = W
                self.T = generator2stochmat(Q)
                print("GENERATOR: generator constructed from arbitrary weight matrix")
        else:
            print("GENERATOR: explicit generator provided")
        # record variables
        self.Q = Q
        self.n_state = self.Q.shape[0]
        self.jump_rate = jump_rate
        self.forward = forward
        self.symmetrize = symmetrize
        self.process_generator()
        # for plotting
        self.set_target_axis()

    def process_generator(self, Q=None, check=True):
        if Q is not None:
            self.Q = Q

        if check:
            self._check_generator()

        # eigen_decompositions
        evals_fwd, EVEC_fwd = eigen_decomp(
            self.Q, right=True
        )  # propagates forward in time
        evals_bwd, EVEC_bwd = eigen_decomp(
            self.Q, right=False
        )  # propagates backward in time
        self.EVEC_fwd = EVEC_fwd
        self.EVECinv_fwd = LA.inv(EVEC_fwd)
        self.evals_fwd = evals_fwd
        self.EVEC_bwd = EVEC_bwd
        self.EVECinv_bwd = LA.inv(EVEC_bwd)
        self.evals_bwd = evals_bwd
        evals_info(self.evals_fwd)

        # polar coordinates
        self.eradians_fwd = np.angle(self.evals_fwd)
        self.edegrees_fwd = np.angle(self.evals_fwd, deg=True)
        self.eradii_fwd = np.abs(evals_fwd)
        self.eradians_bwd = np.angle(self.evals_bwd)
        self.edegrees_bwd = np.angle(self.evals_bwd, deg=True)
        self.eradii_bwd = np.abs(evals_bwd)

    @property
    def evals(self):
        return self.evals_fwd

    @property
    def EVEC(self):
        return self.EVEC_fwd

    @property
    def EVECinv(self):
        return self.EVECinv_fwd

    @property
    def eradians(self):
        return self.eradians_fwd

    @property
    def edegrees(self):
        return self.edegrees_fwd

    @property
    def eradii(self):
        return self.eradii_fwd

    def is_unitary(self):
        """ checks if generator is a unitary matrix """
        check = is_unitary(self.Q)
        if check:
            print('Generator is unitary.')
        else:
            print('Generator is not unitary.')

    def is_normal(self):
        """checks if generator is a normal matrix """
        check = is_normal(self.Q)
        if check:
            print('Generator is normal.')
        else:
            print('Generator is non-normal.')

    def state_spectral_weights(self, state):
        """ returns w[k] = \phi_k(state)*\phi^inv_k[state] """
        w = np.zeros(self.n_state)
        for k in range(self.n_state):
            w[k] = self.EVEC[state, k] * self.EVECinv[k, state]
        return w

    def spectral_matrix(self):
        """ returns W[k,i,j] = \phi_k(state_i)*\phi^inv_k[state_j] """
        W = np.zeros((self.n_state, self.n_state, self.n_state))
        for k in range(self.n_state):
            W[k, :, :] = np.outer(self.EVEC[:, k], self.EVECinv[k, :])
        return W

    def highlight_states(self, states, weight=100):
        """ manipulates jump rate at particular states to over-represent in superdiffusive sequence generation"""
        if not hasattr(states, "len"):
            states = [states]
        for state in states:
            self.Q[state, :] /= weight
        self.process_generator()

    def stationary_dist(self):
        """ returns an estimate of the stationary distribution based on the top spectral components """
        rho_inf = self.EVEC[:, 0] ** 2
        # rho_inf = self.EVEC[:,0]*self.EVECinv[:,0]
        assert np.allclose(
            rho_inf.sum(), 1
        ), "stationary distribution estimate does not sum to 1"
        return rho_inf

    def compose_gen_correct(self):
        """
        Compose generators based on higher-order corrections.
        Check if commutative.
        """
        print("todo")

    def compose_gen_conj(self):
        """
        Compose generators based on conjunction.
        """
        print("todo")

    def _check_generator(self):
        """
        Checks whether Q is a generator.
        """
        self.Q = check_generator(self.Q, symmetrize=self.symmetrize)

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

    def fig_axis(self, figsize=(10, 10)):
        """
        FUNCTION: Sets figure and/or axis to plot to.
        """
        if self.no_target_axis:
            fig, ax = plt.subplots(figsize=figsize)
            self.fig = fig
            self.ax = ax
        else:
            plt.axes(self.ax)

    def plot_generator_matrix(self):
        """Plots generator matrix to self.ax."""
        self.fig_axis(figsize=(12, 10))
        # set color range to have 0 = white
        vmin = self.Q.min()
        vmax = self.Q.max()
        vmin = np.min([vmin, -vmax])
        vmax = np.max([-vmin, vmax])
        im = self.ax.imshow(
            self.Q, origin="upper", cmap=plt.cm.bwr, vmin=vmin, vmax=vmax
        )
        # self.ax.axis('equal')
        plt.colorbar(im, shrink=1)
        plt.ylabel("current state")
        plt.xlabel("future state")
        # remove_axes(self.ax)
        plt.grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_eigenspectrum(self, frame="cartesian"):
        """Plot eigenvalues to self.ax in cartesian or polar coordinates."""
        self.fig_axis()
        if frame == "cartesian":
            # cartesian coordinates
            self.ax.plot(self.evals.real, "-o", label="Real", zorder=99)
            self.ax.plot(self.evals.imag, "-o", label="Imaginary", zorder=98)
            self.ax.set_xlabel("Eigenvalue order")
            self.ax.set_ylabel("Eigenvalue")
            self.ax.set_xlim([0, self.evals.shape[0]])
            # self.ax.set_ylim([self.evals.min(),self.evals.max()])
            self.ax.legend()
            self.ax.set_title("Eigenspectrum")
            sb.despine(ax=self.ax, top=True, right=True, left=False, bottom=False)
        elif frame == "polar":
            self.ax.scatter(self.edegrees, self.eradii, c="k", zorder=99)
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(360)
            self.ax.set_xlabel("Eigenvalue order")
            self.ax.set_ylabel("Eigenvalue")
            self.ax.set_xlim([0, self.evals.shape[0]])
            self.ax.set_title("Eigenspectrum")

    def plot_real_eigenvectors(
        self,
        kernels=None,
        start=1,
        n=4,
        step=1,
        wrap_col=2,
        title=True,
        norm_scale=True,
    ):
        """
        FUNCTION: Plots real components of eigenvectors in state-space.
        INPUTS: kernels = kernel numbers to plot
                start = kernel to start with
                n = how many eigenvectors to plot (skipping the "top" zero eigenvector)
                step = number of e-vectors to skip between plotted e-vectors
                wrap_col = maximum number of columns
                title = whether to set titles
                norm_scale = whether to plot all e-vectors on same color scale
        """
        if kernels is None:
            kernels = range(start, n + start, step)
        else:
            n = len(kernels)
        # max/min intensity values
        vmin = np.real(self.EVEC).min()
        vmax = np.real(self.EVEC).max()
        gridspec_kw["wspace"] = 0.2
        if self.no_target_axis:
            nrow, ncol = grid_wrap_nrow(total=n, wrap_col=wrap_col)
            self.fig, self.axs = plt.subplots(
                nrows=nrow,
                ncols=ncol,
                gridspec_kw=gridspec_kw,
                sharex=False,
                sharey=True,
                figsize=(3*page_width, nrow * row_height),
            )
            self.axs = np.asarray(self.axs).reshape(-1)
        else:
            self.axs = [self.ax]

        # plot eigenvectors
        for ix, evec_ix in enumerate(kernels):
            ax = self.axs[ix]
            if np.any(np.iscomplex(self.EVEC[:, evec_ix])):
                raise ValueError("plot_eigenvectors: complex eigenvector!")
            else:
                evec = self.EVEC[:, evec_ix].real
            if norm_scale:
                ax = self.ENV.plot_state_func(
                    state_vals=evec, ax=ax, cmap=cmap_grid_code, vmin=vmin, vmax=vmax
                )
            else:
                ax = self.ENV.plot_state_func(
                    state_vals=evec, ax=ax, cmap=cmap_grid_code
                )
            evali = self.evals[evec_ix]
            if title:
                ax.set_title("e-vec #%i | e-val = %.2f" % (evec_ix, evali))
        # delete unneeded axes
        if self.no_target_axis:
            for ix in range(n, nrow * ncol):
                self.fig.delaxes(self.axs[ix])
            plt.tight_layout()

    def plot_eigenvectors(self, n=6):
        """
        FUNCTION: Plots eigenvectors (both real and imaginary components) in state-space.
        INPUTS: n = how many eigenvectors to plot (skipping the "top" zero eigenvector)
        """
        # max/min intensity values
        vmin = np.min([self.EVEC.real.min(), self.EVEC.imag.min()])
        vmax = np.min([self.EVEC.real.max(), self.EVEC.imag.max()])
        self.fig, self.axs = plt.subplots(
            nrows=n,
            ncols=2,
            gridspec_kw=gridspec_kw,
            sharex=False,
            sharey=True,
            figsize=[vis.page_width, vis.row_height * n],
        )
        self.axs = self.axs.reshape(-1)

        # plot eigenvectors
        for ix in range(n):
            evec_ix = ix + 1  # skip top eigenvector
            ax_real = self.axs[2 * ix]
            ax_imag = self.axs[2 * ix + 1]
            evec_real = self.EVEC[:, evec_ix].real
            evec_imag = self.EVEC[:, evec_ix].imag
            self.ENV.plot_state_func(
                state_vals=evec_real,
                ax=ax_real,
                cmap=cmap_grid_code,
                vmin=vmin,
                vmax=vmax,
            )
            self.ENV.plot_state_func(
                state_vals=evec_imag,
                ax=ax_imag,
                cmap=cmap_grid_code,
                vmin=vmin,
                vmax=vmax,
            )
            evali = self.evals[evec_ix]
            ax_real.set_title("e-vector %i, real" % evec_ix)
            ax_imag.set_title("e-vector %i, imag" % evec_ix)
        plt.tight_layout()

    def plot_jump_rates(self, ax=None):
        """
        FUNCTION: plot jump rates of generator on state-space.
        """
        self.fig_axis()
        self.ENV.plot_state_func(state_vals=np.diag(self.Q), ax=self.ax, cbar=False)
        self.ax.set_title("Limit density for long time steps")


@timeit_debug
def stochmat2generator(T, jump_rate=10.0):
    """
    Returns the CTMC generator defined by the embedded DTMC T and
    jump intensity jump_rate (equiv. time constant).
    T            = DTMC stochastic matrix
    jump_rate    = jump intensity parameter (higher is more jumps)
    """
    assert np.allclose(T.sum(1), 1), "rows of T do not sum to 1"
    Q = jump_rate * (T - np.eye(T.shape[0]))
    # check_generator(Q)
    return Q


@timeit_debug
def weightmat2generator(W, normsym=True):
    """
    FUCNTION:
        Returns the CTMC generator defined by graph weights W (can be negative).
    INPUTS:
        W       = weight matrix for graph
        normsym = returns normalized symmetric graph Laplacian, otherwise standard W generator
    Equals symmetric normalized graph Laplacian.

    """
    check_weightmat(W)
    if normsym:
        Q = -symnorm_graph_laplacian(W)
        Q = set_generator_diagonal(Q)
    else:
        Q = set_generator_diagonal(W)
    # check_generator(Q)
    return Q


def adjmat2generator(A, jump_rate=10.0):
    """
    Returns the CTMC generator defined by diffusion on graph with weighted
    adjacency matrix A and jump intensity jump_rate (equiv. time constant).
    A           = weighted adjacency matrix for graph
    jump_rate    = jump intensity parameter (somewhat redundant so defaults to 1)
    """
    assert (
        np.diag(A) == 0
    ).all(), "Adjacency matrix should be zero on the diagonal (no self-adjacencies)"
    Q = jump_rate * A.astype("float")
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q

def adjmat2stochmat(A):
    """
    Returns the stochastic matrix associated with adjacency matrix A
    A = weighted adjacency matrix for graph
    """
    assert (
        np.diag(A) == 0
    ).all(), "Adjacency matrix should be zero on the diagonal (no self-adjacencies)"
    return A/A.sum(1, keepdims=True)


def potential2generator(Psi, W, jump_rate=10.0, beta=1.0):
    """
    Returns the infinitesimal generator of gradient flow defined by the ATTRACTIVE potential Psi.
    Psi     = potential
    W       = weight matrix
    beta    = inverse temperature (gradient noise)
    """
    n_state = W.shape[0]
    A = np.abs(W.astype("float"))  # weighted adjacency matrix
    QPsi = np.zeros(W.shape)
    for s1 in range(n_state):
        for s2 in range(n_state):
            QPsi[s1, s2] = np.clip(
                jump_rate * beta * W[s1, s2] * (Psi[s2] - Psi[s1]), a_min=0, a_max=None
            )
    np.fill_diagonal(QPsi, -QPsi.sum(axis=1))
    Qbeta = adjmat2generator(A=A, jump_rate=jump_rate, beta=beta)
    check_generator(Qbeta)
    check_generator(QPsi)
    check_generator(QPsi + Qbeta)
    return {"Q": QPsi + Qbeta, "QPsi": QPsi, "Qbeta": Qbeta}


def generator2weightmat(Q):
    W = deepcopy(Q.astype("float"))
    W[np.eye(W.shape[0], dtype=bool)] = 0.0
    return W


@timeit_debug
def generator2stochmat(Q, tau=0.0, zero_diag=True):
    """
    FUNCTION: CTMC generator to DTMC transition matrix.
    INPUTS: Q           = generator
            tau         = prior on transition probability
            zero_diag   = zero out diagonal
    """
    T = Q.astype("float").copy()
    if zero_diag:
        T[np.eye(T.shape[0]).astype("bool")] = 0
    else:
        jump_rate = np.diagonal(T)
        T = T / jump_rate + np.eye(T.shape)
    T = row_norm(T)
    T = row_norm(T + tau)
    return T


def symmetrize_generator(Q):
    Qsym = (Q + Q.T) / 2.0
    Qsym = set_generator_diagonal(Qsym)
    return Qsym


def set_generator_diagonal(Q):
    Q[np.eye(Q.shape[0], dtype=bool)] = 0.0
    # Q = np.round(Q,10)
    for i in range(Q.shape[0]):
        Q[i, i] = -np.sum(Q[i, :])
    # Q[np.abs(Q)<10**-10] = 0.
    return Q


def symmetrize_weightmat(W):
    """
    FUNCTION: symmetrizes weight matrix.
    """
    print("todo")


def process_potential(Psi, Pstart, mass=0.0):
    if mass != 0.0 or Psi is not None:
        if Psi is None:
            Psi = -mass * Pstart
        else:
            Psi = Psi - mass * Pstart
    return Psi


def ensure_generator(Q):
    """
    FUNCTION: Applies constraints necessary to ensure that Q is a generator matrix.
    """
    n_state = Q.shape[0]
    diag_ix = np.eye(n_state).astype("bool")
    offdiag_ix = (~diag_ix) & (Q != 0)
    Qout = deepcopy(Q)

    error = Q.sum(axis=1)
    if np.any(error != 0):
        "ensure_generator: making sure Q is a generator."
        for state_ix in range(n_state):
            trans_ix = np.where(offdiag_ix[state_ix, :])[0]  # find nonzero transitions
            Qout[state_ix, trans_ix] = (
                Q[state_ix, trans_ix] - error[state_ix] / trans_ix.size
            )
    check_generator(Qout)
    return Qout


def check_generator(Q, symmetrize=False):
    """
    Checks whether Q is a generator.
    """
    is_gen = True
    if not np.allclose(Q.sum(1), 0):
        print("GENERATOR: matrix rows do not sum to 0.")
        is_gen = False
    else:
        print("GENERATOR: matrix rows sum to 0.")
    if np.any(Q[~np.eye(Q.shape[0], dtype=bool)] < 0.0):
        print("GENERATOR: some matrix off-diagonals are negative.")
        is_gen = False
    if np.any(np.diag(Q) > 0.0):
        print("GENERATOR: some matrix diagonals are non-negative.")
        is_gen = False
    if is_symmetric(Q):
        print("GENERATOR: generator is symmetric.")
    else:
        print("GENERATOR: generator is not symmetric.")
        if symmetrize:
            Q = symmetrize_generator(Q)
            assert is_symmetric(Q)
            print("GENERATOR: generator symmetrized.")
    if is_gen:
        print("GENERATOR: Q is a generator with shape", Q.shape, ".")
    else:
        raise ValueError("GENERATOR: Q is not a generator.")

    return Q

def evals_info(evals):
    """Prints eigenvalue information."""
    print(
        "EIGENSPECTRUM: algebraic multiplicity of zero eigenvalue =",
        np.sum(evals == 0.0),
    )
    if np.unique(evals).size != evals.size:
        unique, counts = np.unique(evals, return_counts=True)
        comb = np.vstack((unique[counts > 1], counts[counts > 1])).T
        print("EIGENSPECTRUM: algebraic multiplicity > 1.")
    if LA.norm(evals[np.iscomplex(evals)]) > config.min_val:
        print("EIGENSPECTRUM: complex eigenvalues:", evals[np.iscomplex(evals)])
    if np.any(np.real(evals) > 0.0):
        print("EIGENSPECTRUM: real components of eigenvalues in positive domain:")
        print((evals[evals > 0.0]))


def check_weightmat(W):
    """
    Checks whether W is a weight matrix.
    """
    if not np.all(np.diag(W) == 0):
        raise ValueError("WEIGHT MATRIX: nonzero on the diagonal.")
    if not is_symmetric(W):
        raise ValueError("WEIGHT MATRIX: not symmetric.")


def modify_jump_rates(Q, states, jr=10):
    """ modifies the jumprate at specific states
        jr must be positive, higher values defines faster jumps """
    for b in states:
        Q[b, b] = -jr
    for i in range(Q.shape[0]):
        total = Q[i, :].sum()
        if total != 0:
            offdiags = np.where(Q[i, :] != 0)[0]
            offdiags = np.array([s for s in offdiags if s != i])
            Q[i, offdiags] = Q[i, offdiags] - total / offdiags.size
    return Q
