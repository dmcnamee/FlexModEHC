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
from visualization import save_figure, label_panels, label_panel, page_width, row_height, color_diff, color_superdiff, color_acmin
from scipy.stats import sem


run_explorer = True
run_learner = True
run_sampler = True
resample = True
save_output = True


figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures'))
simdir = os.path.abspath(os.path.join(os.getcwd(), 'simulations'))
ensure_dir(figdir)
ensure_dir(simdir)


# SETTINGS - ENVIRONMENT
start = None
jump_rate = 15. # 15.

# SETTINGS - GENERATOR

# SETTINGS - PROPAGATOR
rho_init = 'start' # minimize autocorrelation wrt 'start' or 'stationary' distribution
lags_opt = [1,2,3,4,5,6,7,8,9]
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
kwargs = {'jitter_std':0.02, 'state_msize':20, 'state_lw':0.5, 'traj_width':0.5, 'traj_format':'-o', 'color_time':True, 'cmap_samp':'husl'}
alpha = 0.2

# autotransition = 0.5
tau_diff = 20.7
alpha_diff = 1.
tau_supdiff = 3.1
alpha_supdiff = 0.3


objs = ['T', 'SR']
obj = 'SR_corr'

# simulation
n_workers = 50
n_step_exp = 100
n_step_exp_min = 100
n_samp_exp = 1
n_workers_exp = n_workers
n_step_lrn = 50
n_samp_lrn = 500
n_workers_lrn = n_workers
l_percentage = 1
n_step_sam = 10
n_step_sam_min = 10
n_samp_sam = 10
n_workers_sam = n_workers



discount = 0.9 # MDP horizon discount
lr = 0.3
lr_decay = 0.999

# ENVIRONMENT
n_clique = 5; n_state_clique = 10
ENV = RingOfCliques(start=start, n_clique=n_clique, n_state_clique=n_state_clique)
ENV.draw_graph(with_labels=True)

states_bneck = [np.arange(i*n_state_clique,i*n_state_clique+2) for i in range(n_clique)]
states_cliques = [np.arange(i*n_state_clique,(i+1)*n_state_clique) for i in range(n_clique)]
states_internal = [s for s in range(n_clique*n_state_clique) if not s in np.concatenate(states_bneck)]

# anti-clockwise directionality
from utils import row_norm
eps = 0.2
for c in range(n_clique-1):
    state_bneck_out = states_bneck[c][1]
    state_bneck_in = states_bneck[c+1][0]
    states_clique = [s for s in states_cliques[c] if s != state_bneck_out]
    ENV.T[states_clique,state_bneck_out] = 1.
    ENV.T[state_bneck_out,state_bneck_in] = 1.
state_bneck_out = states_bneck[-1][1]
state_bneck_in = states_bneck[0][0]
states_clique = [s for s in states_cliques[-1] if s != state_bneck_out]
ENV.T[states_clique,state_bneck_out] = 1.
ENV.T[state_bneck_out,state_bneck_in] = 1.
ENV.T[(ENV.T<1)&(ENV.T>0)] = eps
ENV.T = row_norm(ENV.T)
ENV.__name__ += '-anticlockwise'

# %%
GEN = Generator(ENV=ENV, jump_rate=jump_rate)
PROPd = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROPs = Propagator(GEN=GEN, tau=tau_supdiff, alpha=alpha_supdiff)
PROPo = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROPo.min_zero_cf(lags=lags_opt, rho_init=rho_init)

print('DIFF: average autotransition prob = %0.3f'%np.diag(PROPd.etO).mean())
print('SUPDIFF: average autotransition prob = %0.3f'%np.diag(PROPs.etO).mean())


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
        EXPd = Explorer(PROP=PROPd, rho_init=rho_init_exp, no_dwell=no_dwell_exp, label='DIFF_w%i_%i_%i'%(worker,n_step_exp,n_samp_exp), target_dir=simdir)
        EXPs = Explorer(PROP=PROPs, rho_init=rho_init_exp, no_dwell=no_dwell_exp, label='SUPERDIFF_w%i_%i_%i'%(worker,n_step_exp,n_samp_exp), target_dir=simdir)
        EXPo = Explorer(PROP=PROPo, rho_init=rho_init_exp, no_dwell=no_dwell_exp, label='ACMIN_w%i_%i_%i'%(worker,n_step_exp,n_samp_exp), target_dir=simdir)
        EXPd.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPd.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_exp)
        EXPs.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPs.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_exp)
        EXPo.sample_sequences(n_samp=n_samp_exp, n_step=n_step_exp)
        EXPo.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_exp)
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
        LRNd = Learner(PROP=PROPd, rho_init=rho_init_lrn, discount=discount, no_dwell=no_dwell_lrn, label='DIFF_w%i_%i_%i'%(worker,n_step_lrn,n_samp_lrn), target_dir=simdir)
        LRNs = Learner(PROP=PROPs, rho_init=rho_init_lrn, discount=discount, no_dwell=no_dwell_lrn, label='SUPERDIFF_w%i_%i_%i'%(worker,n_step_lrn,n_samp_lrn), target_dir=simdir)
        LRNo = Learner(PROP=PROPo, rho_init=rho_init_lrn, discount=discount, no_dwell=no_dwell_lrn, label='ACMIN_w%i_%i_%i'%(worker,n_step_lrn,n_samp_lrn), target_dir=simdir)
        LRNd.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)
        LRNs.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)
        LRNo.sample_sequences(n_samp=n_samp_lrn, n_step=n_step_lrn)

        LRNd.learn_cumulative_sample(lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs)
        LRNs.learn_cumulative_sample(lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs)
        LRNo.learn_cumulative_sample(lr=lr, lr_decay=lr_decay, percentage=l_percentage, objs=objs)

        loss_d = LRNd._retrieve_learn_var(n_samp=None, key=obj).values
        loss_s = LRNs._retrieve_learn_var(n_samp=None, key=obj).values
        loss_o = LRNo._retrieve_learn_var(n_samp=None, key=obj).values
        losses_d.append(loss_d)
        losses_s.append(loss_s)
        losses_o.append(loss_o)

    losses_d = np.array(losses_d) # run x #samples matrix
    losses_s = np.array(losses_s) # run x #samples matrix
    losses_o = np.array(losses_o) # run x #samples matrix


if run_sampler:
    cov_samples_s = []
    cov_samples_d = []
    cov_samples_o = []
    for worker in range(n_workers_exp):
        SAMd = Explorer(PROP=PROPd, rho_init=rho_init_sam, no_dwell=no_dwell_sam, label='DIFF_w%i_%i_%i'%(worker,n_step_sam,n_samp_sam), target_dir=simdir)
        SAMs = Explorer(PROP=PROPs, rho_init=rho_init_sam, no_dwell=no_dwell_sam, label='SUPERDIFF_w%i_%i_%i'%(worker,n_step_sam,n_samp_sam), target_dir=simdir)
        SAMo = Explorer(PROP=PROPo, rho_init=rho_init_sam, no_dwell=no_dwell_sam, label='ACMIN_w%i_%i_%i'%(worker,n_step_sam,n_samp_sam), target_dir=simdir)
        SAMd.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMd.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_sam)
        SAMs.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMs.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_sam)
        SAMo.sample_sequences(n_samp=n_samp_sam, n_step=n_step_sam)
        SAMo.compute_diagnostics(embedded_space=embedded_space, flight_vision=flight_vision_sam)
        # sequence-based analysis
        cov_samples_s.append(SAMs.coverage_samples)
        cov_samples_d.append(SAMd.coverage_samples)
        cov_samples_o.append(SAMo.coverage_samples)

    cov_samples_s = np.array(cov_samples_s)
    cov_samples_d = np.array(cov_samples_d)
    cov_samples_o = np.array(cov_samples_o)



# %% PLOT
width = page_width
height = row_height*2
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(width, height))
ax0 = axes[0][0]
ax1 = axes[0][1]
ax2 = axes[0][2]
ax3 = axes[1][0]
ax4 = axes[1][1]
ax5 = axes[1][2]

# state-space panel
EXPd.ENV.plot_environment(ax=ax0)
ax0.margins(0.05, tight=True)

# learning/consolidation panel
n_consol_acc_steps = 12

samp_iter_lrn = range(n_step_lrn+1)
losses_d_mean = losses_d.mean(0)
losses_s_mean = losses_s.mean(0)
losses_o_mean = losses_o.mean(0)
losses_d_sem = sem(losses_d, 0)
losses_s_sem = sem(losses_s, 0)
losses_o_sem = sem(losses_o, 0)
ax1.plot(LRNd.n_samp_learncum[:n_consol_acc_steps], losses_d_mean[:n_consol_acc_steps], '-o', color=color_diff, label='diffusion', clip_on=True, zorder=100)
ax1.fill_between(LRNd.n_samp_learncum[:n_consol_acc_steps], losses_d_mean[:n_consol_acc_steps] - losses_d_sem[:n_consol_acc_steps], losses_d_mean[:n_consol_acc_steps] + losses_d_sem[:n_consol_acc_steps], color=color_diff, alpha=alpha, clip_on=True, zorder=100)
ax1.scatter(LRNd.n_samp_learncum[n_consol_acc_steps-1], losses_d_mean[n_consol_acc_steps-1], color=color_diff, clip_on=False)

ax1.plot(LRNs.n_samp_learncum[:n_consol_acc_steps], losses_s_mean[:n_consol_acc_steps], '-o', color=color_superdiff, label='superdiffusion', clip_on=True, zorder=100)
ax1.fill_between(LRNs.n_samp_learncum[:n_consol_acc_steps], losses_s_mean[:n_consol_acc_steps] - losses_s_sem[:n_consol_acc_steps], losses_s_mean[:n_consol_acc_steps] + losses_s_sem[:n_consol_acc_steps], color=color_superdiff, alpha=alpha, clip_on=True, zorder=100)
ax1.scatter(LRNs.n_samp_learncum[n_consol_acc_steps-1], losses_s_mean[n_consol_acc_steps-1], color=color_superdiff, clip_on=False)

ax1.plot(LRNo.n_samp_learncum[:n_consol_acc_steps], losses_o_mean[:n_consol_acc_steps], '-o', color=color_acmin, label='min-autocorr', clip_on=True, zorder=100)
ax1.fill_between(LRNo.n_samp_learncum[:n_consol_acc_steps], losses_o_mean[:n_consol_acc_steps] - losses_o_sem[:n_consol_acc_steps], losses_o_mean[:n_consol_acc_steps] + losses_o_sem[:n_consol_acc_steps], color=color_acmin, alpha=alpha, clip_on=True, zorder=100)
ax1.scatter(LRNo.n_samp_learncum[n_consol_acc_steps-1], losses_o_mean[n_consol_acc_steps-1], color=color_acmin, clip_on=False)

ax1.set_xlabel('no. of sequences generated')
ax1.set_ylabel('SR matrix correlation')
ax1.set_title('consolidation accuracy')
ax1.set_xticks([0,10,20,30,40,50])
ax1.set_xlim([0,50])
ax1.set_ylim([0,1])
ax1.text(x=2, y=0.92, s='diffusion', color=color_diff)
ax1.text(x=2, y=0.83, s='superdiffusion', color=color_superdiff)
ax1.text(x=2, y=0.74, s='min-autocorr', color=color_acmin)

vmin = np.min([LRNd.est_SR,LRNs.est_SR,LRNo.est_SR,LRNs.SR])
vmax = np.max([LRNd.est_SR,LRNs.est_SR,LRNo.est_SR,LRNs.SR])
sb.heatmap(LRNd.SR, vmin=vmin, vmax=vmax, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', rasterized=True, cbar=True, cbar_kws=None, cbar_ax=None, square=True, xticklabels='auto', yticklabels='auto', mask=None, ax=ax2)
ax4.imshow(LRNd.est_SR, vmin=vmin, vmax=vmax)
ax3.imshow(LRNs.est_SR, vmin=vmin, vmax=vmax)
ax5.imshow(LRNo.est_SR, vmin=vmin, vmax=vmax)

for ax in [ax2,ax3,ax4,ax5]:
    ax.set_xlabel('future state')
    ax.set_ylabel('initial state')
    ax.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
sb.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax3, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax4, top=True, right=True, left=True, bottom=True)
sb.despine(ax=ax5, top=True, right=True, left=True, bottom=True)
ax2.set_title('true SR')
ax3.set_title('learned SR (superdiffusion)')
ax4.set_title('learned SR (diffusion)')
ax5.set_title('learned SR (min-autocorr)')

x = -0.25
y = 1.3
label_panel(ax0, label='A', x=x+0.25, y=y)
label_panel(ax1, label='B', x=x, y=y)
label_panel(ax2, label='C', x=x, y=y)
label_panel(ax3, label='D', x=x+0.14, y=y)
label_panel(ax4, label='E', x=x-0.14, y=y)
label_panel(ax5, label='F', x=x-0.09, y=y)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.6, hspace=0.8)
fig.set_size_inches(width, height)

if save_output:
    fname_base = 'FIGURE_S6'
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')





# %% SUPPLEMENTARY PLOT
width = page_width*1.3
height = row_height*1.6*2.5
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
axes[0][c].set_title('diffusion', pad=title_pad, fontsize=title_fontsize)

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
axes[0][c].set_title('superdiffusion', pad=title_pad, fontsize=title_fontsize)

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
axes[0][c].set_title('min-autocorrelation', pad=title_pad, fontsize=title_fontsize)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.tight_layout()

if save_output:
    fname_base = 'FIGURE_S5'
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
