#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

from pprint import pprint
from scipy.stats import entropy, spearmanr
from numpy.linalg import norm
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy

import config
from utils import SR, to_one_hot
from sr_dyna import (
    update_reward,
    update_transition_matrix,
    update_sr
)
from rl_graph import optimal_policy, successor_rep
from simulators import Simulator
from generators import (
    check_generator,
    set_generator_diagonal,
    stochmat2generator,
    generator2stochmat,
)
from utils import row_norm
from numba import jit, prange
from timer import timeit_debug, timeit_info
from visualization import save_figure


class Learner(Simulator):
    """Simulator class with additional structure learning facilities."""

    @timeit_info
    def __init__(self, PROP, label="LEARNER", beta_sftmx=1.0, discount=0.99, **kwargs):
        super(Learner, self).__init__(PROP=PROP, label=label, **kwargs)
        self.error_curves_computed = False
        self.beta_sftmx = beta_sftmx
        self.discount = discount
        self.SR = SR(self.GEN.T, gamma=discount)

    @timeit_debug
    def _set_file_names(self):
        super(Learner, self)._set_file_names()
        self.fname_objs = os.path.join(
            self.output_dir, "objs_" + self.fname_base + ".csv"
        )

    @timeit_debug
    def _set_output_container(self):
        """
        FUNCTION: Establishes pandas dataframes for storing results.
        NOTES: output_simulation records scalar simulation characteristics for every trajectory sample/step
               output_diagnostics records scalar exploration diagnostics for every step
        """
        super(Learner, self)._set_output_container()
        learnIX = pd.Index(data=range(self.n_samp + 1), name="n_samp")
        self.output_learn = pd.DataFrame(
            data=np.nan,
            index=learnIX,
            columns=[
                "alpha",
                "tau",
                "lr",
                "Q_error",
                "Q_corr",
                "T_error",
                "T_corr",
                "SR_error",
                "SR_corr",
                "R_error",
                "SPL_error",
                "policy_error",
                "KLtraj",
            ],
            dtype="float",
        )

    @timeit_debug
    def save_output(self):
        """Saves output in "flat" format for plotting etc"""
        super(Learner, self).save_output()
        output_learn = self.output_learn.reset_index()  # flatten
        output_learn.to_csv(self.fname_learn)  # save

    @timeit_debug
    def _record_learn_var(self, n_samp, **kwargs):
        """Record loss variable in self.output_learn"""
        for key, value in kwargs.items():
            self.output_learn.loc[n_samp, key] = value

    @timeit_debug
    def _retrieve_learn_var(self, n_samp, key="Q_error"):
        """
        FUNCTION: Retrieve loss variable in self.output_learn
        INPUT: n_samp   = trajectory sample size
               key      = column in self.output_learn to return
        """
        if n_samp is None:
            ix = self.ix_slice[self.n_samp_learncum]
        else:
            ix = n_samp
        return self.output_learn.loc[ix, key]

    @timeit_debug
    def estimate_transition_matrix(self, lr=0.3, samps=None, prior=None, weight=1.0):
        """
        FUNCTION: estimates a transition matrix from sample paths.
        INPUTS: lr = 0.3, learning rate
                samps  = samples to use
                prior   = prior transition matrix
                weight  = weight on prior in estimation
        OUTPUTS: self.est_T = estimated transition matrix
        NOTES: could use sr_dyna.update_transition_matrix instead
        """
        self.lr_T = lr
        if samps is None:
            samps = range(self.n_samp)
        # n_samp x n_seq_steps matrix of sequence samples
        state_seqs = self._retrieve_state(samp=None, step=None, coords=False)
        if not hasattr(self, "est_T"):
            est_T = np.eye(self.n_state)
        else:
            est_T = self.est_T

        for s, samp in enumerate(samps):
            traj = state_seqs[samp, :]
            lr = self.lr_T * (self.lr_decay ** s)
            for state_ix, state_ix_next in zip(traj[:-1], traj[1:]):
                est_T[state_ix, state_ix_next] = est_T[state_ix, state_ix_next] + lr * (
                    1.0 - est_T[state_ix, state_ix_next]
                )
        if prior is not None:
            est_T = est_T + weight * prior
        self.est_T = est_T
        self.est_A = (self.est_T > 0).astype("int")  # estimated adjacency matrix

    @timeit_debug
    def estimate_generator(
        self, lr=0.3, samps=None, prior=None, rand_undiscovered=False
    ):
        """
        FUNCTION: Estimates a transition matrix and generator from sequence samples under propagator PROP.
        INPUTS: lr = 0.3, learning rate
                samps = samples to use
                prior = transition matrix prior
                rand_undiscovered = True, randomizes unobserved transitions (to all states)
                                    otherwise removes such states from the state index
        OUTPUTS: self.est_Q = estimated generator matrix
                 self.est_T = estimated transition matrix
        """
        if samps is None:
            samps = range(self.n_samp)
        jr = self.GEN.jump_rate

        if not hasattr(self, "est_T"):
            est_T = np.eye(self.n_state)
            # est_T = row_norm(np.ones((self.n_state,self.n_state)))
            est_Q = np.ones((self.n_state, self.n_state))
            est_Q = set_generator_diagonal(est_Q)
        else:
            est_T = self.est_T
            est_Q = self.est_Q

        if samps is not None and len(samps) > 0:
            # n_samp x n_seq_steps matrix of sequence samples
            state_seqs = self._retrieve_state(samp=samps, step=None, coords=False)

            st = state_seqs.flatten()
            st_pairs = list(zip(st, st[1:]))
            self.estimate_transition_matrix(lr=lr, samps=samps, prior=prior)

            # deal with undiscovered states and convert to generator
            undiscovered_states = np.all(est_T == 0, axis=1)
            if rand_undiscovered is True:
                # randomize unobserved states
                est_T[undiscovered_states, :] = 1.0
                est_T = row_norm(est_T)
                est_Q = stochmat2generator(est_T, jump_rate=jr)
            else:
                # isolate undiscovered states
                est_T[np.ix_(undiscovered_states, undiscovered_states)] = 1.0
                est_T = row_norm(est_T)
                est_Q = stochmat2generator(est_T, jump_rate=jr)
                est_Q[undiscovered_states, :] = 0.0

        est_Q = set_generator_diagonal(est_Q)
        self.est_T = est_T
        self.est_Q = est_Q
        self.est_A = (self.est_T > 0).astype("int")  # estimated adjacency matrix

    @timeit_debug
    def learn_SR(self, lr=0.3, discount=None, samps=None):
        """
        FUNCTION: Estimates the successor representation from sequence samples under propagator PROP.
        INPUTS: lr           = learning rate
                discount     = temporal discount
                samps        = samples to use
        OUTPUTS: self.est_SR = estimated successor representation matrix
        """
        if discount is None:
            discount = self.discount
        if not hasattr(self, "est_SR"):
            T_prior = row_norm(np.ones((self.n_state, self.n_state)))
            est_SR = SR(T_prior, gamma=discount)
        else:
            est_SR = self.est_SR
        if samps is None:
            samps = range(self.n_samp)
        self.lr_SR = lr
        self.discount_est_SR = discount
        for s in samps:
            lr = self.lr_SR * (self.lr_decay ** s)
            state_traj = self._retrieve_state(samp=s, step=None, coords=False)
            est_SR = update_sr(est_SR, state_traj, discount=discount, learning_rate=lr)
        self.est_SR = est_SR
        self.SR_error = self._obj_norm(L=self.est_SR, T=self.SR, normalized=False)
        self.SR_corr, _ = spearmanr(self.est_SR.flatten(), self.SR.flatten())
        if config.verbose:
            print("LEARNER: SR error = %.3f" % self.SR_error)

    @timeit_debug
    def learn_reward(self, lr=0.1, samps=None):
        """
        FUNCTION: Estimates expected reward
        INPUTS: lr          = learning rate
                samps        = samples to use
        OUTPUTS: self.est_R = estimated expected reward
                self.R_error = error in estimated expected reward
        """
        assert hasattr(
            self.ENV, "R"
        ), "Environment does not have a reward function associated with it."
        assert (
            not self.output_scalar.loc[self.ix_slice[0:n_samp, :], "reward"]
            .isnull()
            .any()
        ), "Missing rewards."
        if not hasattr(self, "est_R"):
            est_R = np.zeros(self.n_state)
        else:
            est_R = self.est_R
        if samps is None:
            samps = range(self.n_samp)
        self.lr_R = lr
        assert n_samp <= self.n_samp, "Too many samples requested."
        for s in samps:
            lr = self.lr_R * (self.lr_decay ** s)
            state_traj = self._retrieve_state(samp=s, step=None, coords=False)
            reward_traj = self._retrieve_reward(
                samp=s, step=None
            )  # FIXME reward_traj nans
            est_R = update_reward(
                reward=est_R,
                state_sequence=state_traj,
                reward_observed=reward_traj,
                alpha=lr,
            )
        self.est_R = est_R
        self.R_error = self._obj_norm(
            L=self.est_R, T=self.ENV.R_state, normalized=False
        )  # compute reward objective
        if config.verbose:
            print("LEARNER: reward error = %.3f" % self.R_error)

    @timeit_info
    def learn(self, lr=0.1, lr_decay=1.0, discount=None, samps=None, objs=["T"]):
        """
        FUNCTION: Learns/estimates SR/T/Q/R.
        INPUTS: lr          = learning rate
                lr_decay    = decay rate of learning rate
                discount    = temporal discount
                samps        = samples to use
                objs        = ['Q', 'T', 'SR', 'KLtraj', 'SPL', 'R'] objectives to learn
        """
        if discount is None:
            discount = self.discount
        self.lr = lr
        self.lr_decay = lr_decay
        self.objs = objs
        if "SR" in objs:
            self.learn_SR(lr=lr, discount=discount, samps=samps)
        if "Q" in objs or "T" in objs:
            self.estimate_generator(lr=lr, samps=samps)
            self.Q_error = self._obj_norm(L=self.est_Q, T=self.GEN.Q)
            self.T_error = self._obj_norm(L=self.est_T, T=self.GEN.T)
            self.T_corr, _ = spearmanr(self.est_T.flatten(), self.GEN.T.flatten())
            self.Q_corr, _ = spearmanr(self.est_Q.flatten(), self.GEN.Q.flatten())
            if config.verbose:
                print("LEARNER: Generator error = %.3f" % self.Q_error)
        if "KLtraj" in objs:
            self.KLtraj = self._obj_KLtraj(Tlearn=self.est_T, Ttarget=self.GEN.T)
            if config.verbose:
                print("LEARNER: Generator trajectory KL = %.3f" % self.KLtraj)
        if "SPL" in objs:
            self.SPL_error = self._obj_distance_norm(L=self.est_Q, T=self.GEN.Q)
            if config.verbose:
                print("LEARNER: Shortest distance error = %.3f" % self.SPL_error)
        if hasattr(self.ENV, "R") and "R" in objs:
            self.learn_reward(lr=lr, samps=samps)
            self.policy_error = self._obj_policy()

    @timeit_debug
    def learn_cumulative_sample(
        self, lr=0.1, lr_decay=1.0, discount=None, percentage=20, objs=["T", "KLtraj"]
    ):
        """
        FUNCTION: Learns as a function of number of sequence samples.
        INPUTS: lr          = learning rate
                lr_decay    = decay rate of learning rate
                discount    = temporal discount
                percentage  = percentage of sample numbers to learn cumulatively from
                              i.e. 10 means that each learning pass will use another 10% of the samples
                n_samp      = number of sequences to use
                objs        = ['Q', 'T', 'SR', 'KLtraj', 'SPL', 'R'] objectives to learn
        """
        if discount is None:
            discount = self.discount
        assert hasattr(self, "n_samp"), "Trajectories not sampled."
        self.l_samp_rate = int(np.max([1, self.n_samp * (percentage / 100.0)]))
        self.n_samp_learncum = np.arange(0, self.n_samp, self.l_samp_rate).astype("int")
        self.n_samp_learncum = np.concatenate((np.array([0]), self.n_samp_learncum))

        # record prior errors
        samps = np.array([])
        self.learn(lr=lr, lr_decay=lr_decay, discount=discount, samps=samps, objs=objs)
        if "Q" in objs:
            self._record_learn_var(n_samp=0, Q_error=self.Q_error)
            self._record_learn_var(n_samp=0, Q_corr=self.T_corr)
        if "T" in objs:
            self._record_learn_var(n_samp=0, T_error=self.T_error)
            self._record_learn_var(n_samp=0, T_corr=self.T_corr)
        if "SR" in objs:
            self._record_learn_var(n_samp=0, SR_error=self.SR_error)
            self._record_learn_var(n_samp=0, SR_corr=self.SR_corr)
        if "KLtraj" in objs:
            self._record_learn_var(n_samp=0, KLtraj=self.KLtraj)
        if "SPL" in objs:
            self._record_learn_var(n_samp=0, SPL_error=self.SPL_error)
        if hasattr(self.ENV, "R") and "R" in objs:
            self._record_learn_var(n_samp=0, R_error=self.R_error)
            self._record_learn_var(n_samp=0, policy_error=self.policy_error)

        for ix in range(self.n_samp_learncum.size - 1):
            samps = self.ix_samps[
                self.n_samp_learncum[ix] : self.n_samp_learncum[ix + 1]
            ]
            n_samp_learncum = self.n_samp_learncum[ix + 1]
            self.learn(
                lr=lr, lr_decay=lr_decay, discount=discount, samps=samps, objs=objs
            )
            if "Q" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, Q_error=self.Q_error)
                self._record_learn_var(n_samp=n_samp_learncum, Q_corr=self.Q_corr)
            if "T" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, T_error=self.T_error)
                self._record_learn_var(n_samp=n_samp_learncum, T_corr=self.T_corr)
            if "SR" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, SR_error=self.SR_error)
                self._record_learn_var(n_samp=n_samp_learncum, SR_corr=self.SR_corr)
            if "KLtraj" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, KLtraj=self.KLtraj)
            if "SPL" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, SPL_error=self.SPL_error)
            if hasattr(self.ENV, "R") and "R" in objs:
                self._record_learn_var(n_samp=n_samp_learncum, R_error=self.R_error)
                self._record_learn_var(
                    n_samp=n_samp_learncum, policy_error=self.policy_error
                )
        # record key parameters
        self.output_learn.loc[:, "alpha"] = self.PROP.alpha
        self.output_learn.loc[:, "tau"] = self.PROP.tau
        self.output_learn.loc[:, "lr"] = self.lr
        self.error_curves_computed = True

    @timeit_debug
    def plot_learning_errors(self, objs=["T_error"], figdir=None):
        """
        FUNCTION: Plots learning objectives as a function of number of sequence samples.
        INPUTS:
        objs = ['Q_error', 'Q_corr', 'T_error', 'T_corr', 'SR_error', 'SR_corr', 'SPL_error', 'KLtraj', 'R_error', 'policy_error'] objectives to learn
        [Superseded by plot_performance_metrics]
        """
        assert self.error_curves_computed, "Learning curves not yet computed."
        self.fig_axis()
        ax = self.ax
        for key in objs:
            loss = self._retrieve_learn_var(n_samp=None, key=key).values
            ax.plot(range(1, self.n_samp + 1), loss, "--o", label=key)
        ax.set_ylabel("Euclidean error")
        ax.set_xlabel("#samples")
        ax.legend(loc="center left")
        if figdir is not None:
            save_figure(
                fig=self.fig,
                figdir=figdir,
                fname_base="learning_analysis",
                file_ext=".png",
            )

    @timeit_debug
    def plot_performance_metrics(
        self, metrics=["Q_error", "SR_error", "SPL_error", "policy_error"], figdir=None
    ):
        """
        FUNCTION: Plots learning objectives on different panels as a function of number of sequences.
        INPUTS: metrics = list of what metrics to plot from self.metrics_dict.keys()
        """
        self.metrics_dict = {
            "Q_error": "generator error",
            "Q_corr": "generator correlation",
            "T_error": "transition error",
            "T_corr": "transition correlation",
            "SR_error": "successor representation error",
            "SR_corr": "successor representation correlation",
            "R_error": "reward function error",
            "SPL_error": "shortest path length error",
            "policy_error": "expected cumulative reward loss",
            "KLtraj": "trajectory KL-divergence",
        }
        metrics_df = pd.melt(
            self.output_learn.reset_index(),
            id_vars=["n_samp", "alpha", "tau", "lr"],
            value_vars=self.metrics_dict.keys(),
            var_name="metric",
            value_name="value",
        )
        metrics_df.rename(columns={"n_samp": "#samples"}, inplace=True)
        for metric_key in self.metrics_dict.keys():
            if metric_key in metrics:
                metrics_df.loc[
                    metrics_df.metric == metric_key, "metric"
                ] = self.metrics_dict[metric_key]
            else:
                metrics_df = metrics_df[metrics_df.metric != metric_key]
        self.metrics_learn = metrics_df
        g = sb.FacetGrid(
            self.metrics_learn,
            row="lr",
            col="metric",
            hue="alpha",
            margin_titles=True,
            sharex=True,
            sharey=False,
        )
        g.map(
            sb.regplot,
            "#samples",
            "value",
            ci=None,
            fit_reg=False,
            logx=False,
            order=1,
            truncate=False,
        )
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        if figdir is not None:
            save_figure(
                fig=g, figdir=figdir, fname_base="learning_performance", file_ext=".png"
            )

    @timeit_debug
    def plot_KLtraj_curve(self, normalized=True):
        """Plots KL trajectory learning objective as a function of number of sequence samples."""
        assert self.error_curves_computed, "Learning curves not yet computed."
        self.fig_axis()
        ax = self.ax
        key = "KLtraj"
        loss = self._retrieve_learn_var(n_samp=None, key=key).values
        ax.plot(np.arange(1, self.n_samp + 1), loss, "-o", label=key, color="black")
        ax.set_ylabel("Trajectory KL loss (nats)")
        ax.set_xlabel("Number of sequence samples")
        ax.legend(loc="center left")

    def plot_dyn_mat(self, ax=None, type="T", learned=True, vmin=None, vmax=None):
        """
        imshow plots either est_SR / est_T / est_Q depending on 'type' and 'learned'
        """
        self.set_target_axis(ax)
        if type is "T":
            if learned:
                ax.imshow(self.est_T, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(self.T, vmin=vmin, vmax=vmax)
        elif type is "SR":
            if learned:
                ax.imshow(self.est_SR, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(self.SR, vmin=vmin, vmax=vmax)
        elif type is "Q":
            if learned:
                ax.imshow(self.est_Q, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(self.GEN.Q, vmin=vmin, vmax=vmax)
        else:
            raise ValueError("unrecognized learned dynamics matrix request")
        self.ax.set_xlabel("future state")
        self.ax.set_ylabel("initial state")
        self.ax.tick_params(
            axis="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        sb.despine(ax=self.ax, top=True, right=True, left=True, bottom=True)

    def mds_dyn_mat(self, ax=None, type="T", learned=True):
        """
        dimensionality reduction plot for either est_SR or est_T depending on 'type' and 'learned'
        """
        import networkx as nx
        from sklearn.manifold import MDS, SpectralEmbedding

        dr = SpectralEmbedding(
            n_components=2,
            affinity="precomputed",
            gamma=None,
            random_state=None,
            eigen_solver=None,
            n_neighbors=3,
            n_jobs=None,
        )

        self.set_target_axis(ax)
        if type is "T":
            if learned:
                S = (self.est_T + self.est_T.T) / 2.0
            else:
                S = (self.ENV.T + self.ENV.T.T) / 2.0
        elif type is "SR":
            if learned:
                S = (self.est_SR + self.est_SR.T) / 2.0
            else:
                S = (self.SR + self.SR.T) / 2.0
        elif type is "Q":
            if learned:
                S = (self.est_Q + self.est_Q.T) / 2.0
            else:
                S = (self.GEN.Q + self.GEN.Q.T) / 2.0
        else:
            raise ValueError("unrecognized learned dynamics matrix request")

        pos = dr.fit(S).embedding_[:, :2]

        G = nx.from_numpy_array(self.ENV.T)
        pos_dict = dict([(i, posxy) for i, posxy in enumerate(pos)])
        nx.draw(
            G,
            pos=pos_dict,
            ax=self.ax,
            node_size=10,
            node_color="black",
            edge_color="grey",
            width=0.5,
        )

    @timeit_debug
    def _obj_norm(self, L, T, normalized=True):
        """
        FUNCTION: learning objective based on the euclidean norm between L and T (frobenius for matrices).
        INPUTS: L/T = representations to compare (learned/target)
                normalized_
        """
        Ldiag = L.copy()
        Tdiag = T.copy()
        np.fill_diagonal(Ldiag, 0)
        np.fill_diagonal(Tdiag, 0)
        if normalized:
            max_val = np.concatenate((Ldiag, Tdiag)).max()
            Ldiag = Ldiag / max_val
            Tdiag = Tdiag / max_val
        return norm(L - T, ord="fro", axis=None)

    @timeit_debug
    def _obj_KLtraj(self, Tlearn, Ttarget, tau=0.001, normalized=True):
        """
        FUNCTION: learning objective based on the KL-divergence between trajectory distributions.
        INPUTS: Tlearn/Ttarget  = generators to compare (learned/target)
                tau             = prior on learned transition matrix (avoids inf in KL due to non-overlapping policy support)
                normalized  = normalizes by number of states
        OUTPUTS: KL[target trajs || learn trajs] normalized by number of states
        """
        # SRlearn = SR(Tlearn)
        SRtarget = SR(Ttarget)

        # KLlocal = np.ones((self.n_state,))
        local_log_diff = np.ones((self.n_state,))
        for n in range(self.n_state):
            if Tlearn[n, :].sum() == 1:
                # KLlocal[n] = entropy(pk=Ttarget[n,:], qk=Tlearn[n,:])
                local_log_diff[n] = np.log(Ttarget[n, :]) - np.log(Tlearn[n, :])

        if normalized:
            return np.dot(SRtarget, local_log_diff).sum() / float(self.n_state)
        else:
            return np.dot(SRtarget, local_log_diff).sum()

    @timeit_debug
    def _obj_distance_norm(self, L, T, normalized=True):
        """
        FUNCTION: frobenius norm between induced shortest distance matrices.
        INPUTS: L/T         = generators to compare (learned/target)
                normalized  = normalizes by number of states
        OUTPUTS: ||D(L)-D(T)||
        """
        GL = nx.Graph(L)
        GT = nx.Graph(T)
        DL = floyd_warshall_numpy(GL)
        DT = floyd_warshall_numpy(GT)
        ix = np.isinf(DT)
        DL[ix] = 0.0
        DL[np.isinf(DL)] = 0.0
        DT[ix] = 0.0
        distance_norm = norm(DL - DT, ord=None)
        if normalized:
            distance_norm = distance_norm / (self.n_state ** 2)
        return distance_norm

    @timeit_debug
    def compute_policy_target_mdp(self):
        """computes optimal policy given true state-space structure and reward"""
        self.policy_target, self.policy_target_value = optimal_policy(
            adjmat=self.ENV.A_adj,
            reward=self.ENV.R_state,
            discount=self.discount,
            beta_sftmx=self.beta_sftmx,
        )

    @timeit_debug
    def compute_policy_learned_mdp(self):
        """computes optimal policy given estimated state-space structure and reward"""
        self.policy_learn, self.policy_learn_value = optimal_policy(
            adjmat=self.est_A,
            reward=self.est_R,
            discount=self.discount,
            beta_sftmx=self.beta_sftmx,
        )
        # evaluate policy on true reward function
        self.policy_learn_value = np.dot(
            successor_rep(self.policy_learn, self.discount), self.ENV.R_state
        )

    @timeit_debug
    def _obj_policy(self, normalized=True):
        """quantifies loss in expected cumulative reward due to policy error"""
        self.compute_policy_learned_mdp()
        self.compute_policy_target_mdp()
        policy_error = norm(
            self.policy_target_value - self.policy_learn_value, ord=None
        )
        if normalized:
            policy_error = policy_error / self.n_state
        return policy_error
