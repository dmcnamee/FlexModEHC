#!/usr/bin/python
# -*- coding: utf-8 -*-


import os

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from utils import ensure_dir
from environments import RingOfCliques
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from learners import Learner
from visualization import (
    save_figure,
    label_panels,
    label_panel,
    page_width,
    row_height,
    color_diff,
    color_superdiff,
    color_acmin,
)
from scipy.stats import sem


run_visualization = False
run_explorer = True
run_learner = True
run_sampler = True
resample = True
save_workers = False
save_output = True


figdir = os.path.abspath(os.path.join(os.getcwd(), "figures/"))
simdir = os.path.abspath(os.path.join(os.getcwd(), "simulations/"))
ensure_dir(figdir)
ensure_dir(simdir)


# SETTINGS - ENVIRONMENT
start = None
jump_rate = 15.0  # 15.

# SETTINGS - GENERATOR
symmetrize = True  # default True, checked with False

# SETTINGS - PROPAGATOR
rho_init = "start"  # minimize autocorrelation wrt 'start' or 'stationary' distribution
lags_opt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
rho_init_exp = 2
rho_init_lrn = None
rho_init_sam = 2

# SETTINGS - SIMULATOR/LEARNER/EXPLORER
no_dwell_exp = True
no_dwell_lrn = True
no_dwell_sam = False
diagnostics = True
embedded_space = False
flight_vision_exp = True
flight_vision_lrn = False
flight_vision_sam = False

# SETTINGS - VISUALIZATION
kwargs = {
    "jitter_std": 0.02,
    "state_msize": 20,
    "state_lw": 0.5,
    "traj_width": 0.5,
    "traj_format": "-o",
    "color_time": True,
    "cmap_samp": "husl",
}
alpha = 0.2  # (transparency on plots)

# SETTINGS - SIMS

# autotransition ~= 0.5
tau_diff = 20.7
alpha_diff = 1.0
tau_supdiff = 3.1
alpha_supdiff = 0.3


objs = ["T", "SR"]
obj = "SR_corr"  #  SR_corr, SR_error

# full
n_step_exp = 100
n_step_exp_min = 100
n_samp_exp = 1
n_workers_exp = 50
n_step_lrn = 50
n_samp_lrn = 500
n_workers_lrn = 50
l_percentage = 1
n_step_sam = 10
n_step_sam_min = 10
n_samp_sam = 10
n_workers_sam = 50

# fast
# n_step_exp = 10
# n_step_exp_min = 2
# n_samp_exp = 10
# n_workers_exp = 2
# n_step_lrn = 50
# n_samp_lrn = 500
# n_workers_lrn = 50
# l_percentage = 1
# n_step_sam = 10
# n_step_sam_min = 10
# n_samp_sam = 10
# n_workers_sam = 2

discount = 0.9  # MDP horizon discount
lr = 0.3
lr_decay = 0.999

ENV = RingOfCliques(start=start, n_clique=5, n_state_clique=10)
GEN = Generator(ENV=ENV, jump_rate=jump_rate, symmetrize=symmetrize)
PROPd = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROPs = Propagator(GEN=GEN, tau=tau_supdiff, alpha=alpha_supdiff)
PROPo = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROPo.min_zero_cf(lags=lags_opt, rho_init=rho_init)


print("DIFF: average autotransition prob = %0.3f" % np.diag(PROPd.etO).mean())
print("SUPDIFF: average autotransition prob = %0.3f" % np.diag(PROPs.etO).mean())

if run_visualization:
    PROP = PROPd
    ENV.plot_environment()
    ENV.draw_graph()
    EXP = Explorer(PROP=PROP, rho_init=ENV.start, no_dwell=no_dwell_exp)
    EXP.set_viz_scheme(**kwargs)
    EXP.sample_sequences(n_samp=1, n_step=10)

# %% SIMS
if run_explorer:
    exp_eff_s = []
    exp_eff_d = []
    exp_eff_o = []
    cov_visits_s = []
    cov_visits_d = []
    cov_visits_o = []
    traj_cost_sum_s = []
    traj_cost_sum_d = []
    traj_cost_sum_o = []
    n_visits_s = []
    n_visits_d = []
    n_visits_o = []
    for worker in range(n_workers_exp):
        EXPd = Explorer(
            PROP=PROPd,
            rho_init=rho_init_exp,
            no_dwell=no_dwell_exp,
            label="DIFF_w%i_%i_%i" % (worker, n_step_exp, n_samp_exp),
            target_dir=simdir,
        )
        EXPs = Explorer(
            PROP=PROPs,
            rho_init=rho_init_exp,
            no_dwell=no_dwell_exp,
            label="SUPERDIFF_w%i_%i_%i" % (worker, n_step_exp, n_samp_exp),
            target_dir=simdir,
        )
        EXPo = Explorer(
            PROP=PROPo,
            rho_init=rho_init_exp,
            no_dwell=no_dwell_exp,
            label="ACMIN_w%i_%i_%i" % (worker, n_step_exp, n_samp_exp),
            target_dir=simdir,
        )
        EXPd.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPd.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_exp
        )
        EXPs.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPs.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_exp
        )
        EXPo.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPo.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_exp
        )
        # sequence-based analysis
        exp_eff_s.append(EXPs.exp_eff)
        exp_eff_d.append(EXPd.exp_eff)
        exp_eff_o.append(EXPo.exp_eff)

        traj_cost_sum_s.append(EXPs.traj_cost_sum)
        traj_cost_sum_d.append(EXPd.traj_cost_sum)
        traj_cost_sum_o.append(EXPo.traj_cost_sum)
        cov_visits_s.append(EXPs.coverage_visits)
        cov_visits_d.append(EXPd.coverage_visits)
        cov_visits_o.append(EXPo.coverage_visits)
        n_visits_s.append(EXPs.n_distinct_visits)
        n_visits_d.append(EXPd.n_distinct_visits)
        n_visits_o.append(EXPo.n_distinct_visits)

        if save_workers:
            EXPd.save_output()
            EXPd.save()
            EXPs.save_output()
            EXPs.save()
            EXPo.save_output()
            EXPo.save()

    exp_eff_d = np.array(exp_eff_d)
    exp_eff_s = np.array(exp_eff_s)
    exp_eff_o = np.array(exp_eff_o)
    traj_cost_sum_s = np.array(traj_cost_sum_s)
    traj_cost_sum_d = np.array(traj_cost_sum_d)
    traj_cost_sum_o = np.array(traj_cost_sum_o)
    cov_visits_s = np.array(cov_visits_s)
    cov_visits_d = np.array(cov_visits_d)
    cov_visits_o = np.array(cov_visits_o)
    n_visits_s = np.array(n_visits_s)
    n_visits_d = np.array(n_visits_d)
    n_visits_o = np.array(n_visits_o)

if run_learner:
    losses_d = []
    losses_s = []
    losses_o = []
    for worker in range(n_workers_lrn):
        LRNd = Learner(
            PROP=PROPd,
            rho_init=rho_init_lrn,
            discount=discount,
            no_dwell=no_dwell_lrn,
            label="DIFF_w%i_%i_%i" % (worker, n_step_lrn, n_samp_lrn),
            target_dir=simdir,
        )
        LRNs = Learner(
            PROP=PROPs,
            rho_init=rho_init_lrn,
            discount=discount,
            no_dwell=no_dwell_lrn,
            label="SUPERDIFF_w%i_%i_%i" % (worker, n_step_lrn, n_samp_lrn),
            target_dir=simdir,
        )
        LRNo = Learner(
            PROP=PROPo,
            rho_init=rho_init_lrn,
            discount=discount,
            no_dwell=no_dwell_lrn,
            label="ACMIN_w%i_%i_%i" % (worker, n_step_lrn, n_samp_lrn),
            target_dir=simdir,
        )
        LRNd.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)
        LRNs.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)
        LRNo.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)

        LRNd.learn_cumulative_sample(
            lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs
        )
        LRNs.learn_cumulative_sample(
            lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs
        )
        LRNo.learn_cumulative_sample(
            lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs
        )

        loss_d = LRNd._retrieve_learn_var(n_samp=None, key=obj).values
        loss_s = LRNs._retrieve_learn_var(n_samp=None, key=obj).values
        loss_o = LRNo._retrieve_learn_var(n_samp=None, key=obj).values
        losses_d.append(loss_d)
        losses_s.append(loss_s)
        losses_o.append(loss_o)

    LRNd.save_output()
    LRNd.save()
    LRNs.save_output()
    LRNs.save()
    LRNo.save_output()
    LRNo.save()
    losses_d = np.array(losses_d)  #  run x #samples matrix
    losses_s = np.array(losses_s)  #  run x #samples matrix
    losses_o = np.array(losses_o)  #  run x #samples matrix



if run_sampler:
    cov_samples_s = []
    cov_samples_d = []
    cov_samples_o = []
    for worker in range(n_workers_exp):
        SAMd = Explorer(
            PROP=PROPd,
            rho_init=rho_init_sam,
            no_dwell=no_dwell_sam,
            label="DIFF_w%i_%i_%i" % (worker, n_step_sam, n_samp_sam),
            target_dir=simdir,
        )
        SAMs = Explorer(
            PROP=PROPs,
            rho_init=rho_init_sam,
            no_dwell=no_dwell_sam,
            label="SUPERDIFF_w%i_%i_%i" % (worker, n_step_sam, n_samp_sam),
            target_dir=simdir,
        )
        SAMo = Explorer(
            PROP=PROPo,
            rho_init=rho_init_sam,
            no_dwell=no_dwell_sam,
            label="ACMIN_w%i_%i_%i" % (worker, n_step_sam, n_samp_sam),
            target_dir=simdir,
        )
        SAMd.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMd.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_sam
        )
        SAMs.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMs.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_sam
        )
        SAMo.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMo.compute_diagnostics(
            embedded_space=embedded_space, flight_vision=flight_vision_sam
        )
        # sequence-based analysis
        cov_samples_s.append(SAMs.coverage_samples)
        cov_samples_d.append(SAMd.coverage_samples)
        cov_samples_o.append(SAMo.coverage_samples)

        if save_workers:
            SAMd.save_output()
            SAMd.save()
            SAMs.save_output()
            SAMs.save()
            SAMo.save_output()
            SAMo.save()
    cov_samples_s = np.array(cov_samples_s)
    cov_samples_d = np.array(cov_samples_d)
    cov_samples_o = np.array(cov_samples_o)


# %% PLOT
# width = page_width * 1.3
# height = row_height * 1.6
# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(width, height))

# extended figure with MDS
width = page_width * 1.3
height = row_height * 2.4
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(width, height))

ax2 = axes[0][0]
ax0 = axes[0][1]
ax1 = axes[0][2]
ax3 = axes[0][3]


# state-space panel
EXPd.ENV.plot_environment(ax=ax2)
ax2.margins(0.05, tight=True)

# exploration efficiency panel
include_hw = False
first_iter = 0

samp_iter_exp = range(n_step_exp_min + 1)
samp_iter_exp = samp_iter_exp[first_iter:]
exp_eff_d_mean = exp_eff_d.mean(0)[first_iter:]
exp_eff_s_mean = exp_eff_s.mean(0)[first_iter:]
exp_eff_o_mean = exp_eff_o.mean(0)[first_iter:]
exp_eff_d_sem = sem(exp_eff_d, 0)[first_iter:]
exp_eff_s_sem = sem(exp_eff_s, 0)[first_iter:]
exp_eff_o_sem = sem(exp_eff_o, 0)[first_iter:]

traj_cost_sum_d_mean = traj_cost_sum_d.mean(0)[first_iter:]
traj_cost_sum_s_mean = traj_cost_sum_s.mean(0)[first_iter:]
traj_cost_sum_o_mean = traj_cost_sum_o.mean(0)[first_iter:]

cov_visits_d_mean = cov_visits_d.mean(0)[first_iter:]
cov_visits_s_mean = cov_visits_s.mean(0)[first_iter:]
cov_visits_o_mean = cov_visits_o.mean(0)[first_iter:]
cov_visits_d_sem = sem(cov_visits_d, 0)[first_iter:]
cov_visits_s_sem = sem(cov_visits_s, 0)[first_iter:]
cov_visits_o_sem = sem(cov_visits_o, 0)[first_iter:]
n_visits_d_mean = n_visits_d.mean(0)[first_iter:]
n_visits_s_mean = n_visits_s.mean(0)[first_iter:]
n_visits_o_mean = n_visits_o.mean(0)[first_iter:]
n_visits_d_sem = sem(n_visits_d, 0)[first_iter:]
n_visits_s_sem = sem(n_visits_s, 0)[first_iter:]
n_visits_o_sem = sem(n_visits_o, 0)[first_iter:]

max_dist = 200
ix_d = traj_cost_sum_d_mean <= max_dist
ix_s = traj_cost_sum_s_mean <= max_dist
ix_o = traj_cost_sum_o_mean <= max_dist
ax0.plot(
    traj_cost_sum_d_mean[ix_d],
    cov_visits_d_mean[ix_d],
    "-o",
    color=color_diff,
    label="diffusion",
    clip_on=False,
    zorder=102,
)
ax0.plot(
    traj_cost_sum_s_mean[ix_s],
    cov_visits_s_mean[ix_s],
    "-o",
    color=color_superdiff,
    label="superdiffusion",
    clip_on=False,
    zorder=101,
)
ax0.plot(
    traj_cost_sum_o_mean[ix_o],
    cov_visits_o_mean[ix_o],
    "-o",
    color=color_acmin,
    label="min-autocorr",
    clip_on=False,
    zorder=100,
)

ax0.fill_between(
    traj_cost_sum_d_mean[ix_d],
    cov_visits_d_mean[ix_d] - cov_visits_d_sem[ix_d],
    cov_visits_d_mean[ix_d] + cov_visits_d_sem[ix_d],
    color=color_diff,
    alpha=alpha,
    clip_on=False,
    zorder=102,
)
ax0.fill_between(
    traj_cost_sum_s_mean[ix_s],
    cov_visits_s_mean[ix_s] - cov_visits_s_sem[ix_s],
    cov_visits_s_mean[ix_s] + cov_visits_s_sem[ix_s],
    color=color_superdiff,
    alpha=alpha,
    clip_on=False,
    zorder=101,
)
ax0.fill_between(
    traj_cost_sum_o_mean[ix_o],
    cov_visits_o_mean[ix_o] - cov_visits_o_sem[ix_o],
    cov_visits_o_mean[ix_o] + cov_visits_o_sem[ix_o],
    color=color_acmin,
    alpha=alpha,
    clip_on=False,
    zorder=100,
)
ax0.set_xlim([0, max_dist])
ax0.set_ylim([0, 1])
ax0.set_xlabel("no. of steps taken")
ax0.set_ylabel("fraction of states")
ax0.set_title("exploration efficiency")

ax0.text(x=5, y=0.92, s="diffusion", color=color_diff)
ax0.text(x=5, y=0.8, s="superdiffusion", color=color_superdiff)
ax0.text(x=5, y=0.68, s="min-autocorr", color=color_acmin)
if include_hw:
    ax0.plot(
        np.arange(n_step_exp_min + 1),
        np.ones((n_step_exp_min + 1,)),
        "-o",
        c="k",
        clip_on=False,
        zorder=100,
    )
    ax0.text(x=12.2, y=0.85, s="hamiltonian walk", color="k")


# learning/consolidation panel
n_consol_acc_steps = 12

samp_iter_lrn = range(n_step_lrn + 1)
losses_d_mean = losses_d.mean(0)
losses_s_mean = losses_s.mean(0)
losses_o_mean = losses_o.mean(0)
losses_d_sem = sem(losses_d, 0)
losses_s_sem = sem(losses_s, 0)
losses_o_sem = sem(losses_o, 0)
ax1.plot(
    LRNd.n_samp_learncum[:n_consol_acc_steps],
    losses_d_mean[:n_consol_acc_steps],
    "-o",
    color=color_diff,
    label="diffusion",
    clip_on=True,
    zorder=100,
)
ax1.fill_between(
    LRNd.n_samp_learncum[:n_consol_acc_steps],
    losses_d_mean[:n_consol_acc_steps] - losses_d_sem[:n_consol_acc_steps],
    losses_d_mean[:n_consol_acc_steps] + losses_d_sem[:n_consol_acc_steps],
    color=color_diff,
    alpha=alpha,
    clip_on=True,
    zorder=100,
)
ax1.scatter(
    LRNd.n_samp_learncum[n_consol_acc_steps - 1],
    losses_d_mean[n_consol_acc_steps - 1],
    color=color_diff,
    clip_on=False,
)

ax1.plot(
    LRNs.n_samp_learncum[:n_consol_acc_steps],
    losses_s_mean[:n_consol_acc_steps],
    "-o",
    color=color_superdiff,
    label="superdiffusion",
    clip_on=True,
    zorder=100,
)
ax1.fill_between(
    LRNs.n_samp_learncum[:n_consol_acc_steps],
    losses_s_mean[:n_consol_acc_steps] - losses_s_sem[:n_consol_acc_steps],
    losses_s_mean[:n_consol_acc_steps] + losses_s_sem[:n_consol_acc_steps],
    color=color_superdiff,
    alpha=alpha,
    clip_on=True,
    zorder=100,
)
ax1.scatter(
    LRNs.n_samp_learncum[n_consol_acc_steps - 1],
    losses_s_mean[n_consol_acc_steps - 1],
    color=color_superdiff,
    clip_on=False,
)

ax1.plot(
    LRNo.n_samp_learncum[:n_consol_acc_steps],
    losses_o_mean[:n_consol_acc_steps],
    "-o",
    color=color_acmin,
    label="min-autocorr",
    clip_on=True,
    zorder=100,
)
ax1.fill_between(
    LRNo.n_samp_learncum[:n_consol_acc_steps],
    losses_o_mean[:n_consol_acc_steps] - losses_o_sem[:n_consol_acc_steps],
    losses_o_mean[:n_consol_acc_steps] + losses_o_sem[:n_consol_acc_steps],
    color=color_acmin,
    alpha=alpha,
    clip_on=True,
    zorder=100,
)
ax1.scatter(
    LRNo.n_samp_learncum[n_consol_acc_steps - 1],
    losses_o_mean[n_consol_acc_steps - 1],
    color=color_acmin,
    clip_on=False,
)

ax1.set_xlabel("no. of sequences generated")
if obj == "SR_error":
    ax1.set_ylabel("SR estimation error")
elif obj == "SR_corr":
    ax1.set_ylabel("SR matrix correlation")
elif obj == "T_error":
    ax1.set_ylabel("transition estimation error")
elif obj == "T_corr":
    ax1.set_ylabel("transition matrix correlation")
elif obj == "KLtraj":
    ax1.set_ylabel("trajectory distribution")
else:
    raise ValueError("unknown objective")
ax1.set_title("consolidation accuracy")
ax1.set_xticks([0, 10, 20, 30, 40, 50])
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 1])


# coverage panel
samp_iter_sam = range(n_step_sam_min + 1)
cov_samples_d_mean = cov_samples_d.mean(0)
cov_samples_s_mean = cov_samples_s.mean(0)
cov_samples_o_mean = cov_samples_o.mean(0)
cov_samples_d_sem = sem(cov_samples_d, 0)
cov_samples_s_sem = sem(cov_samples_s, 0)
cov_samples_o_sem = sem(cov_samples_o, 0)
ax3.plot(
    samp_iter_sam,
    cov_samples_d_mean,
    "-o",
    color=color_diff,
    label="diffusion",
    clip_on=False,
    zorder=100,
)
ax3.plot(
    samp_iter_sam,
    cov_samples_s_mean,
    "-o",
    color=color_superdiff,
    label="superdiffusion",
    clip_on=False,
    zorder=100,
)
ax3.plot(
    samp_iter_sam,
    cov_samples_o_mean,
    "-o",
    color=color_acmin,
    label="min-autocorr",
    clip_on=False,
    zorder=100,
)
ax3.fill_between(
    samp_iter_sam,
    cov_samples_d_mean - cov_samples_d_sem,
    cov_samples_d_mean + cov_samples_d_sem,
    color=color_diff,
    alpha=alpha,
    clip_on=False,
    zorder=100,
)
ax3.fill_between(
    samp_iter_sam,
    cov_samples_s_mean - cov_samples_s_sem,
    cov_samples_s_mean + cov_samples_s_sem,
    color=color_superdiff,
    alpha=alpha,
    clip_on=False,
    zorder=100,
)
ax3.fill_between(
    samp_iter_sam,
    cov_samples_o_mean - cov_samples_o_sem,
    cov_samples_o_mean + cov_samples_o_sem,
    color=color_acmin,
    alpha=alpha,
    clip_on=False,
    zorder=100,
)

ax3.set_xlabel("no. of sampling iterations")
ax3.set_ylabel("fraction of states")
ax3.set_title("sampling coverage")
ax3.set_xticks(range(n_step_sam + 1))
ax3.set_xlim([0, n_step_sam])
ax3.set_ylim([0, 1])



ax0 = axes[1][2]
ax1 = axes[1][1]
ax3 = axes[1][3]
ax2 = axes[1][0]

if obj == "SR_error" or obj == "SR_corr":
    vmin = np.min([LRNd.est_SR, LRNs.est_SR, LRNo.est_SR, LRNs.SR])
    vmax = np.max([LRNd.est_SR, LRNs.est_SR, LRNo.est_SR, LRNs.SR])
    sb.heatmap(
        LRNd.SR,
        vmin=vmin,
        vmax=vmax,
        center=None,
        robust=False,
        annot=None,
        fmt=".2g",
        annot_kws=None,
        linewidths=0,
        linecolor="white",
        rasterized=True,
        cbar=True,
        cbar_kws=None,
        cbar_ax=None,
        square=True,
        xticklabels="auto",
        yticklabels="auto",
        mask=None,
        ax=ax2,
    )
    LRNd.plot_dyn_mat(ax=ax0, type="SR", vmin=vmin, vmax=vmax)
    LRNs.plot_dyn_mat(ax=ax1, type="SR", vmin=vmin, vmax=vmax)
    LRNo.plot_dyn_mat(ax=ax3, type="SR", vmin=vmin, vmax=vmax)
else:
    vmin = np.min([LRNd.est_T, LRNs.est_T, LRNo.est_T, GEN.T])
    vmax = np.max([LRNd.est_T, LRNs.est_T, LRNo.est_T, GEN.T])
    ax2.imshow(GEN.T, vmin=vmin, vmax=vmax)
    LRNd.plot_dyn_mat(ax=ax0, type="T", vmin=vmin, vmax=vmax)
    LRNs.plot_dyn_mat(ax=ax1, type="T", vmin=vmin, vmax=vmax)
    LRNo.plot_dyn_mat(ax=ax3, type="T", vmin=vmin, vmax=vmax)
for ax in [ax0, ax1, ax2, ax3]:
    ax.set_xlabel("future state")
    ax.set_ylabel("initial state")
    ax.tick_params(
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
sb.despine(ax=ax0, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax3, top=True, right=True, left=True, bottom=True)
ax0.set_title("learned SR (diffusion)")
ax1.set_title("learned SR (superdiffusion)")
ax2.set_title("true SR")
ax3.set_title("learned SR (min-autocorr)")

# SPECTRAL EMBEDDING
ax4 = axes[2][0]
ax5 = axes[2][1]
ax6 = axes[2][2]
ax7 = axes[2][3]

LRNd.mds_dyn_mat(ax=ax4, type="SR", learned=False)
LRNd.mds_dyn_mat(ax=ax6, type="SR")
LRNs.mds_dyn_mat(ax=ax5, type="SR")
LRNo.mds_dyn_mat(ax=ax7, type="SR")


x = -0.25
y = 1.3
label_panel(axes[0][0], label="A", x=x + 0.25, y=y)
label_panel(axes[0][1], label="B", x=x, y=y)
label_panel(axes[0][2], label="C", x=x, y=y)
label_panel(axes[0][3], label="D", x=x, y=y)

label_panel(axes[1][0], label="E", x=x + 0.05, y=y)
label_panel(axes[1][1], label="F", x=x - 0.38, y=y)
label_panel(axes[1][2], label="G", x=x - 0.39, y=y)
label_panel(axes[1][3], label="H", x=x - 0.4, y=y)

label_panel(axes[2][0], label="I", x=x + 0.23, y=y)
label_panel(axes[2][1], label="J", x=x, y=y)
label_panel(axes[2][2], label="K", x=x, y=y)
label_panel(axes[2][3], label="L", x=x, y=y)

fig.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0.6, hspace=0.8)
fig.set_size_inches(width, height)

if save_output:
    fname_base = "FIGURE_7_SamplingOptim_tau%.1f_%.1f_alpha%.1f_%.1f_lr%.1f" % (
        tau_diff,
        tau_supdiff,
        alpha_diff,
        alpha_supdiff,
        lr,
    )
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".png")
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".pdf")


# %% SUPPLEMENTARY PLOT
width = page_width * 1.3
height = row_height * 1.6 * 2.5
nrows = 5
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
state_msize = 5
jitter_state = False
traj_width = 0.5
n_step = 10
rho_start = rho_init_exp
title_fontsize = 24
title_pad = 30

# diffusion
c = 0
for r in range(nrows):
    EXPd.sample_sequences(n_step=n_step, n_samp=1, rho_start=rho_start)
    EXPd.set_target_axis(ax=axes[r][c])
    EXPd.state_msize = state_msize
    EXPd.traj_width = traj_width
    EXPd.jitter_state = jitter_state
    EXPd.color_time = False
    EXPd.color_traj = color_diff
    EXPd.plot_trajectory(plot_env=True, state_func_env=False)
axes[0][c].set_title("diffusion", pad=title_pad, fontsize=title_fontsize)

c = 1
for r in range(nrows):
    EXPs.sample_sequences(n_step=n_step, n_samp=1, rho_start=rho_start)
    EXPs.set_target_axis(ax=axes[r][c])
    EXPs.state_msize = state_msize
    EXPs.traj_width = traj_width
    EXPs.jitter_state = jitter_state
    EXPs.color_time = False
    EXPs.color_traj = color_superdiff
    EXPs.plot_trajectory(plot_env=True, state_func_env=False)
axes[0][c].set_title("superdiffusion", pad=title_pad, fontsize=title_fontsize)

c = 2
for r in range(nrows):
    EXPo.sample_sequences(n_step=n_step, n_samp=1, rho_start=rho_start)
    EXPo.set_target_axis(ax=axes[r][c])
    EXPo.state_msize = state_msize
    EXPo.traj_width = traj_width
    EXPo.jitter_state = jitter_state
    EXPo.color_time = False
    EXPo.color_traj = color_acmin
    EXPo.plot_trajectory(plot_env=True, state_func_env=False)
axes[0][c].set_title("min-autocorrelation", pad=title_pad, fontsize=title_fontsize)

fig.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.tight_layout()

if save_output:
    fname_base = "FIGURE_S5_SamplingOptim_tau%.1f_%.1f_alpha%.1f_%.1f_lr%.1f" % (
        tau_diff,
        tau_supdiff,
        alpha_diff,
        alpha_supdiff,
        lr,
    )
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".png")
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".pdf")
