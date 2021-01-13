#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import numpy as np

from scipy.optimize import minimize
from propagators import Propagator
from timer import timeit_debug, timeit_info
from config import min_val


class Estimator(Propagator):
    @timeit_info
    def __init__(
        self,
        GEN=None,
        tau_true=1.0,
        alpha_true=1.0,
        bounds={"tau": (0.1, 20), "alpha": (0.25, 1.5)},
        dist_mat=None,
        data_seq=None,
        data_reactprob=None,
        label="ESTIMATOR",
        **kwargs
    ):
        """
        Estimator class.
        INPUTS: GEN = generator or dist_mat = distance matrix between states
                tau_true/alpha_true = true regime parameters
                data_seq = (n_samp, n_seq_steps) state sequence samples
                data_reactprob = (n_seq_steps, n_state) state reactivation probabilities
                bounds = dict of tau/alpha bounds
        """
        assert (GEN != None) or (
            dist_mat != None
        ), "Need to define space with generator of distance matrix"
        self.tau_true = tau_true
        self.alpha_true = alpha_true
        if GEN is not None:
            super().__init__(
                GEN=GEN,
                tau=tau_true,
                alpha=alpha_true,
                strict=False,
                label=label,
                **kwargs
            )
        else:
            self.dist_mat = dist_mat
            self.n_state = self.dist_mat.shape[0]
        self.set_data_seq(data=data_seq)
        self.set_data_reactprob(data=data_reactprob)
        self.shift_norm_prop()
        self.bounds = bounds
        self.options = {"disp": True, "maxiter": 1000}

    def set_data_seq(self, data=None):
        self.data_seq = data
        if self.data_seq is not None:
            self.n_samp = self.data_seq.shape[0]
            self.n_seq_steps = self.data_seq.shape[1]

    @timeit_debug
    def _shuffle_along_axis(self, a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    def scramble_data_seq(self):
        """ randomizes data sequences in time """
        map(np.random.shuffle, self.data_seq)
        self.data_seq = self._shuffle_along_axis(self.data_seq, axis=1)

    def set_data_reactprob(self, data=None):
        self.data_reactprob = data
        if self.data_reactprob is not None:
            self.n_seq_steps = self.data_reactprob.shape[0]
            assert (
                self.n_state == self.data_reactprob.shape[1]
            ), "mismatch between state-space definition and reactivation data"

    @timeit_debug
    def NLL(self, return_sum=True):
        """ negative log likelihood that self.data_seq was generated in alpha/tau regime"""
        assert hasattr(self, "data_seq"), "no data"
        # get all state index transition pairs
        pairs = [list(zip(l, l[1:])) for l in self.data_seq]
        pairs = [item for sublist in pairs for item in sublist]
        p = np.array([self.etO[s1, s2] for (s1, s2) in pairs])
        p[p <= 0] = min_val
        if return_sum:
            return -np.log(p).sum()
        else:
            return -np.log(p)

    def NLL_regime(self, tau, alpha):
        """ returns NLL as a function of alpha/tau """
        self.set_new_spec_mod(tau=tau, alpha=alpha)
        return self.NLL()

    def KL(self):
        """ KL-divergence between self.data_reactprob and propagation in the alpha/tau regime """
        assert self.data_reactprob != None, "no data"
        print("todo")

    @timeit_info
    def fit_spectrum_mb(self):
        """ fit spectrum based on environment model encoded in a generator eigendecomposition """
        assert self.data != None, "no data"
        print("todo")

    @timeit_info
    def fit_spectrum_msd(self):
        """ fit spectrum based on msd """
        assert self.data != None, "no data"
        print("todo")

    @timeit_info
    def fit_diff_regime_mb(self):
        """ fit alpha/tau based on environment model encoded in a generator eigendecomposition """
        assert hasattr(self, "data_seq") != None, "no sequence data"
        obj = lambda x: self.NLL_regime(tau=x[0], alpha=x[1])
        self.optres_diff_regime_mb = minimize(
            obj,
            x0=(1.0, 1.0),
            method=None,
            bounds=[self.bounds["tau"], self.bounds["alpha"]],
            tol=None,
            callback=None,
            options=self.options,
        )
        self.tau_mle = self.optres_diff_regime_mb.x[0]
        self.alpha_mle = self.optres_diff_regime_mb.x[1]
        self.nll = self.optres_diff_regime_mb.fun
        self.compute_error()
        return self.optres_diff_regime_mb.x

    @timeit_info
    def fit_diff_regime_msd(self):
        """ fit alpha/tau based on msd """
        assert self.data != None, "no data"
        print("todo")

    @timeit_info
    def NLL_landscape(self, taus, alphas):
        """ compute NLL over tau/alpha grid """
        self.taus = taus
        self.alphas = alphas
        n_tau = len(taus)
        n_alpha = len(alphas)
        self.nlls = np.zeros((n_tau, n_alpha))
        for tau_ix, tau in enumerate(taus):
            for alpha_ix, alpha in enumerate(alphas):
                self.set_new_spec_mod(tau=tau, alpha=alpha)
                self.nlls[tau_ix, alpha_ix] = self.NLL()
        # set mins
        self.tau_mle_ix, self.alpha_mle_ix = np.where(self.nlls == self.nlls.min())
        self.tau_mle = self.taus[self.tau_mle_ix][0]
        self.alpha_mle = self.alphas[self.alpha_mle_ix][0]
        self.compute_error()
        return self.tau_mle, self.alpha_mle

    @timeit_info
    def compute_error(self):
        self.error = np.sqrt(
            (self.tau_true - self.tau_mle) ** 2
            + (self.alpha_true - self.alpha_mle) ** 2
        )
