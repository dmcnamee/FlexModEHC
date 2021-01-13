#!/usr/bin/python
# -*- coding: utf-8 -*-

# {Kay  K. et al. Constant Sub-second Cycling between Representations of Possible Futures in the Hippocampus. \textit{Cell} (2020).}

import os

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from environments import TJunction
from generators import Generator
from propagators import Propagator
from simulators import Simulator
from matplotlib import colors
from visualization import label_panel, page_width, row_height, color_diff, color_superdiff, color_acmin, plot_wfill, save_figure

from numpy.matlib import repmat
from matplotlib.colors import ListedColormap


# VISUALIZATION
save_output = True
cmap_prop = plt.cm.Greys




# %% SETTINGS - ENVIRONMENT
backflow = 0.1 # 0.1
state_start = 23 # 23/26
state_choice = 11
state_junction = 3
states_arm_terminal = [25,27] # terminal states
states_arm_initial = [2,4] # initial arm states
states_central_arm = [26,23,20,17,14,11,8]
states_left_arm = [2,1,0,7,10,13,16,19,22,25]
states_right_arm = [4,5,6,9,12,15,18,21,24,27]
n_arm_states = len(states_central_arm)
blend_weights = np.arange(0,n_arm_states)/(n_arm_states-1)
rho0 = state_start

# SETTINGS - GENERATOR
symmetrize = False
jump_rate = 0.5

# SETTINGS - PROPAGATOR
sigma = 1.
tau = 1. # 1.
alpha_diff = 1
alpha_sdiff = 0.5

# SETTINGS - SIMULATOR/LEARNER/EXPLORER
no_dwell = False
n_samp_plot = 1
n_step_plot = 10
n_samp_est = 100
n_step_est = 20

# SETTINGS - AUTOCORRELATION ANALYSIS
zero_pos = True
ymin = 10**-5
T_opt = 3
lags_opt = [1] # short-horizon -> cycling over shorter horizon
lags_plot = np.arange(1,10)
rho_init = 'start' # 'start', OR: 'uniform', 'stationary'

# SETTINGS - FIGURE
figdir = os.path.join(os.getcwd(), 'figures/')
alpha = 1.
env_lw = 1.
spectrum_colors = np.zeros((n_arm_states,4))
spectrum_colors[0,:] = colors.to_rgba(color_diff)
spectrum_colors[-1,:] = colors.to_rgba(color_acmin)
for ci in range(1,n_arm_states-1):
    spectrum_colors[ci,:] = (1-blend_weights[ci])*spectrum_colors[0,:] + blend_weights[ci]*spectrum_colors[-1,:]
sims_plot = [0,6] # which simulations to plot in panel E




# ENVIRONMENT
ENV = TJunction(start=state_start, directed=True, complete_circuit=False, backflow=backflow)
rho_rand = np.ones((ENV.n_state,))/ENV.n_state
ENV.env_lw = env_lw

# GENERATOR
GEN = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)
rho_inf = GEN.stationary_dist()

# PROPAGATOR
PROP_diff = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_diff)
PROP_sdiff = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_sdiff)
PROP_opt = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_diff)
PROP_opt.min_zero_cf(lags=lags_opt, rho_init=rho_init)
spectrum_blend = repmat(PROP_diff.power_spec, n=1, m=n_arm_states)
spectrum_blend = np.multiply(1-blend_weights.reshape((n_arm_states,1)),spectrum_blend) + np.multiply(blend_weights.reshape((n_arm_states,1)),repmat(PROP_opt.power_spec, n=1, m=n_arm_states))
PROP_blend = []
for i in range(n_arm_states):
    spectrum = spectrum_blend[i,:]
    PROP_blend.append(Propagator(GEN=GEN, power_spec=spectrum))

# SIMULATIONS
SIM_blend = []
for i in range(n_arm_states):
    SIM_blend.append(Simulator(PROP=PROP_blend[i], no_dwell=no_dwell, rho_init=states_central_arm[i]))
SIM_sdiff = Simulator(PROP=PROP_sdiff, no_dwell=no_dwell, rho_init=state_start)

for i in range(n_arm_states):
    SIM_blend[i].sample_sequences(n_samp=n_samp_est, n_step=n_step_est)
SIM_sdiff.sample_sequences(n_samp=n_samp_est, n_step=n_step_est)
SIM_diff = SIM_blend[0]
SIM_opt = SIM_blend[-1]

# AC PREDICTION
Cdiff = PROP_diff.predict_acf(lags=lags_plot, rho_init=rho0)
Csdiff = PROP_sdiff.predict_acf(lags=lags_plot, rho_init=rho0)
Copt = PROP_opt.predict_acf(lags=lags_plot, rho_init=rho0)

# AC ESTIMATION
for i in range(n_arm_states):
    SIM_blend[i].estimate_cf(zero_pos=zero_pos)
SIM_opt.estimate_cf(zero_pos=zero_pos)
SIM_sdiff.estimate_cf(zero_pos=zero_pos)




# %% PLOT
width = page_width*1.2
height = row_height*2
fig, axes = plt.subplots(2, 4, figsize=(width,height), sharex=False, sharey=False)

# DISPLAY TRANSITION FROM LOCALIZATION TO PROSPECTION
state_vals = np.ones((ENV.n_state,))*blend_weights[0]
state_vals[26] = blend_weights[0]
state_vals[23] = blend_weights[1]
state_vals[20] = blend_weights[2]
state_vals[17] = blend_weights[3]
state_vals[14] = blend_weights[4]
state_vals[11] = blend_weights[5]
state_vals[8] = blend_weights[6]
transition_colors = spectrum_colors
transition_colors[:,-1] = alpha
ax0 = axes[0][0]
ENV.plot_state_func(state_vals, ax=ax0, annotate=False, interpolation='nearest', cmap=ListedColormap(transition_colors), cbar=False, cbar_label='', node_edge_color=None, arrows=None, mask_color='white', mask_alpha=0.)
ax0.set_title('shift to planning \n close to decision point', pad=0)

# PLOT SPECTRUMS
ax1 = axes[0][1]
ax1.plot(PROP_diff.power_spec, color=color_diff, label='diffusion', )
ax1.plot(PROP_sdiff.power_spec, color=color_superdiff, label='superdiffusion', zorder=10)
ax1.plot(PROP_opt.power_spec, color=color_acmin, label='min-autocorr')
for i in range(n_arm_states):
    ax1.plot(PROP_blend[i].power_spec, color=spectrum_colors[i,:])
ax1.set_xticks([ENV.n_state-2,1])
ax1.tick_params(axis='x', length=0)
ax1.set_xticklabels(['small', 'large'])
ax1.invert_xaxis()
L = ax1.legend(facecolor='white', framealpha=1, prop={'size': 8})
ax1.axhline(1, linestyle='--', color='k', alpha=0.1, lw=1)
ax1.axhline(-1, linestyle='--', color='k', alpha=0.1, lw=1)
ax1.set_ylim([-1.5,1.25])
ax1.text(x=3.8, y=-1.3, s='DSC')
ax1.set_yticks([-1,-0.5,0,0.5,1])
ax1.set_xlabel(r'spatial scale $ k$')
ax1.set_ylabel(r'power spectrum $ s(k)$')
ax1.set_title('spectral power optimization', pad=23)
sb.despine(ax=ax1, top=True, right=True)

# PLOT SAMPLED SEQUENCES
ax4 = axes[1][0]
state_msize = 5
jitter_std = 0.05
colors = [spectrum_colors[i,:] for i in sims_plot]
# after turn plot
SIM = Simulator(PROP=PROP_blend[0], no_dwell=no_dwell, rho_init=states_arm_initial[0])
SIM.sample_sequences(n_step=n_step_plot, n_samp=1, fast_storage=True)
SIM.set_target_axis(ax4)
SIM.max_steps = n_step_plot
SIM.state_msize = state_msize
SIM.color_time = False
SIM.color_traj = colors[0]
SIM.jitter_std = jitter_std
SIM.plot_trajectory(state_func_env=True, plot_env=True)
# up to turn plots
SIMS = [SIM_blend[i] for i in sims_plot]
colors = [spectrum_colors[i,:] for i in sims_plot]
for ix, SIM in enumerate(SIMS):
    SIM.set_target_axis(ax4)
    SIM.max_steps = n_step_plot
    SIM.state_msize = state_msize
    SIM.color_time = False
    SIM.color_traj = colors[ix]
    SIM.jitter_std = jitter_std
    SIM.plot_trajectory(state_func_env=True, plot_env=False)
ax4.set_title('sampled sequences', pad=1)

# PLOT ESTIMATED AUTOCORRELATIONS
ax5 = axes[1][1]
for ix,SIM in enumerate(SIM_blend):
    plot_wfill(ax=ax5, y=SIM.acf_mean[lags_plot], e=SIM.acf_sem[lags_plot], x=lags_plot, color=spectrum_colors[ix,:], alpha=0.2)
plot_wfill(ax=ax5, y=SIM_sdiff.acf_mean[lags_plot], e=SIM_sdiff.acf_sem[lags_plot], x=lags_plot, color=color_superdiff, alpha=0.2, zorder=10)
ax5.set_xlabel(r'lag $ \Delta t$')
if zero_pos:
    ax5.set_ylabel(r'autocorrelation $ C_X(0,\Delta t)$')
else:
    ax5.set_ylabel(r'autocorrelation $ C_X(t,\Delta t)$')
ax5.set_xticks(lags_plot)
ax5.set_ylim([0, 0.8])
ax5.set_xlim(lags_plot[0],lags_plot[-1])
ax5.set_title('estimated from simulation', pad=17)
sb.despine(ax=ax5, top=True, right=True)

# PLOT PROPAGATORS
# start propagators
vmin = 0
vmax = None
state_central = state_junction; pos_central = [3,0]
pos_initial = [2,0]
ax3 = axes[0][3]

PROP_opt.set_target_axis(ax=ax3)
PROP_opt.plot_prop_kernels(n=1, first_state=state_central, vmin=vmin, vmax=vmax, cmap=cmap_prop)
ax3.set_title('min-autocorr propagator', pad=10)
ax3.text(pos_central[0], pos_central[1], 'S', ha="center", va="center", color="k")


# compare propagation just after choice
ax6 = axes[1][2]
ax7 = axes[1][3]
PROP_diff.set_target_axis(ax=ax6)
PROP_diff.plot_prop_kernels(n=1, first_state=states_arm_initial[0], vmin=vmin, vmax=vmax, cmap=cmap_prop)
ax6.set_title('diffusion propagator', pad=1)
ax6.text(pos_initial[0], pos_initial[1], 'S', ha="center", va="center", color="white")

PROP_opt.set_target_axis(ax=ax7)
PROP_opt.plot_prop_kernels(n=1, first_state=states_arm_initial[0], vmin=vmin, vmax=vmax, cmap=cmap_prop)
ax7.set_title('min-autocorr propagator', pad=1)
ax7.text(pos_initial[0], pos_initial[1], 'S', ha="center", va="center", color="k")

# dominant eigenvector
ax2 = axes[0][2]
GEN.set_target_axis(ax2)
GEN.plot_real_eigenvectors(start=1, n=1)
ax2.set_title('dominant spectral \n component (DSC)', pad=1)


# LABELS, FINESSE AND SAVE
x = -0.25
y = 1.3
label_panel(axes[0][0], 'A', x=x+0.1, y=y)
label_panel(axes[0][1], 'B', x=x-0.1, y=y)
label_panel(axes[0][2], 'C', x=x, y=y)
label_panel(axes[0][3], 'D', x=x, y=y)
label_panel(axes[1][0], 'E', x=x+0.1, y=y-0.1)
label_panel(axes[1][1], 'F', x=x-0.1, y=y-0.1)
label_panel(axes[1][2], 'G', x=x, y=y-0.1)
label_panel(axes[1][3], 'H', x=x, y=y-0.1)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.6, hspace=0.6)
fig.set_size_inches(width, height)

if save_output:
    fname_base = 'FIGURE_5'
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
