#!/usr/bin/python
# -*- coding: utf-8 -*-

# {Pfeiffer  B. E. \& Foster  D. J. Hippocampal place-cell sequences depict future paths to remembered goals. \textit{Nature} \textbf{497}  74-79 (2013).}


import os

import numpy as np
import matplotlib.pyplot as plt

from environments import OpenBox
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from visualization import save_figure, label_panel, page_width, row_height



figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures/'))
fname_base = 'FIGURE_3'
save_output = True


# SETTINGS - ENVIRONMENT
scale = 25 # 25

# SETTINGS - GENERATOR
symmetrize = False
jump_rate = 15
goal_weight = 100

# SETTINGS - PROPAGATOR
tau = 5.
tau_diff = tau
tau_sdiff = tau
alpha_diff = 1.
alpha_sdiff = 0.5
no_dwell = True

# SETTINGS - EXPLORER
n_step_home = 10
n_step_away = 10
n_samp_home = 20
n_samp_away = 10

# VIZ
cmap_samp = 'colorblind'
traj_width = 0.5
density_smooth_sigma = 0.5
cmap_prop = plt.cm.Greys
autoprop_off = True
color_traj = 'black'
color_start = 'red'
color_time = False
cmap_diff = plt.cm.RdBu
traj_width = 0.3
traj_format = '-'
start_pos = True
color_start = 'red'
marker_start = '.'
state_msize = 5.
labelpad = 10
env_lw = 2.
jitter_state = False
vmax_prop = 0.05

# SAMPLING
ENV = OpenBox(scale=scale)
ENV.env_lw = env_lw
state_away_default = 26
states_away = [26, 48, 598, 576]; n_states_away = len(states_away) # corners
state_home = 335 # slightly off center as in Pfeiffer2013
states_home = [state_home]
states_home = [310,311,336,335]


GEN = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)
GENh = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)
GENh.highlight_states(states=states_home, weight=goal_weight)

PROPd = Propagator(GEN=GENh, tau=tau_diff, alpha=alpha_diff)
PROPs = Propagator(GEN=GENh, tau=tau_sdiff, alpha=alpha_sdiff)

# multi-start away explorers
EXPds = []
EXPss = []
PROPds = []
PROPss = []
for start in states_away:
    EXPd = Explorer(PROP=PROPd, rho_init=start, no_dwell=no_dwell, label='diff-start%i'%start)
    EXPs = Explorer(PROP=PROPs, rho_init=start, no_dwell=no_dwell, label='sdiff-start%i'%start)
    PROPds.append(PROPd.etO[start,:])
    PROPss.append(PROPs.etO[start,:])
    EXPd.sample_sequences(n_samp=n_samp_away, n_step=n_step_away)
    EXPs.sample_sequences(n_samp=n_samp_away, n_step=n_step_away)
    EXPds.append(EXPd)
    EXPss.append(EXPs)

# single-start away explorers
EXPda = Explorer(PROP=PROPd, rho_init=state_away_default, no_dwell=no_dwell, label='diff-start%i'%start)
EXPda.sample_sequences(n_samp=n_samp_away, n_step=n_step_away)
EXPsa = Explorer(PROP=PROPs, rho_init=state_away_default, no_dwell=no_dwell, label='sdiff-start%i'%start)
EXPsa.sample_sequences(n_samp=n_samp_away, n_step=n_step_away)

# single-start home explorers
EXPdh = Explorer(PROP=PROPd, rho_init=state_home, no_dwell=no_dwell, label='diff-start%i'%start)
EXPdh.sample_sequences(n_samp=n_samp_home, n_step=n_step_home)
EXPsh = Explorer(PROP=PROPs, rho_init=state_home, no_dwell=no_dwell, label='sdiff-start%i'%start)
EXPsh.sample_sequences(n_samp=n_samp_home, n_step=n_step_home)

# collate multi-start sampling densities
SSd = []
SSs = []
for i in range(n_states_away):
    EXPds[i].compute_sample_density(sigma=density_smooth_sigma)
    SSd.append(EXPds[i].state_seqs_density)
    EXPss[i].compute_sample_density(sigma=density_smooth_sigma)
    SSs.append(EXPss[i].state_seqs_density)
SSd = np.array(SSd).sum(0)
SSs = np.array(SSs).sum(0)
prop_density_da = np.array(PROPds).sum(0)
prop_density_sa = np.array(PROPss).sum(0)

# %% FIGURES
width = page_width*1.1
height = row_height*4
widths = [1,1,1,1]
heights = [1,1,1,1]
fig, axes = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True, gridspec_kw={'width_ratios':widths, 'height_ratios':heights})

# row 0
ax0 = axes[0][0]
ax1 = axes[0][1]
ax2 = axes[0][2]
ax3 = axes[0][3]
ax0.set_axis_off()
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()

# row 1
ax4 = axes[1][0]
ax5 = axes[1][1]
ax6 = axes[1][2]
ax7 = axes[1][3]

# row 2
ax8 = axes[2][0]
ax9 = axes[2][1]
ax10 = axes[2][2]
ax11 = axes[2][3]

# row 3
ax12 = axes[3][0]
ax13 = axes[3][1]
ax14 = axes[3][2]
ax15 = axes[3][3]



# HOME, single-start, "vectorized trajectories"
EXPsh.set_target_axis(ax=ax4)
EXPsh.color_traj = color_traj
EXPsh.cmap_samp = color_traj
EXPsh.color_time = color_time
EXPsh.traj_width = traj_width
EXPsh.traj_format = traj_format
EXPsh.start_pos = start_pos
EXPsh.color_start = color_start
EXPsh.marker_start = marker_start
EXPsh.state_msize = state_msize
EXPsh.jitter_state = jitter_state
EXPsh.plot_trajectories(multi_colored=False)
ax4.set_title('home-events')

EXPdh.set_target_axis(ax=ax8)
EXPdh.color_traj = color_traj
EXPdh.cmap_samp = color_traj
EXPdh.color_time = color_time
EXPdh.traj_width = traj_width
EXPdh.traj_format = traj_format
EXPdh.start_pos = start_pos
EXPdh.color_start = color_start
EXPdh.marker_start = marker_start
EXPdh.state_msize = state_msize
EXPdh.jitter_state = jitter_state
EXPdh.plot_trajectories(multi_colored=False)
ax8.set_title('home-events')

# AWAY, multi-start, "vectorized trajectories"
for i in range(n_states_away):
    EXPs = EXPss[i]
    EXPs.set_target_axis(ax=ax5)
    EXPs.color_traj = color_traj
    EXPs.color_time = color_time
    EXPs.traj_width = traj_width
    EXPs.traj_format = traj_format
    EXPs.start_pos = start_pos
    EXPs.color_start = color_start
    EXPs.marker_start = marker_start
    EXPs.state_msize = state_msize
    EXPs.jitter_state = jitter_state
    EXPs.plot_trajectories(multi_colored=False)
ax5.set_title('away-events')

for i in range(n_states_away):
    EXPd = EXPds[i]
    EXPd.set_target_axis(ax=ax9)
    EXPd.color_traj = color_traj
    EXPd.color_time = color_time
    EXPd.traj_width = traj_width
    EXPd.traj_format = traj_format
    EXPd.start_pos = start_pos
    EXPd.color_start = color_start
    EXPd.marker_start = marker_start
    EXPd.state_msize = state_msize
    EXPd.jitter_state = jitter_state
    EXPd.plot_trajectories(multi_colored=False)
ax9.set_title('away-events')


# HOME, density simulation
EXPsh.set_target_axis(ax=ax6)
EXPdh.set_target_axis(ax=ax10)
EXPsh.compute_sample_density(sigma=density_smooth_sigma)
EXPdh.compute_sample_density(sigma=density_smooth_sigma)
EXPsh.plot_sample_density()
EXPdh.plot_sample_density()
ax6.set_title('sampling density [home]')
ax10.set_title('sampling density [home]')

# AWAY, multi-start, density simulation
EXPsa.set_target_axis(ax=ax7)
EXPda.set_target_axis(ax=ax11)
EXPsa.state_seqs_density = SSs
EXPda.state_seqs_density = SSd
EXPsa.plot_sample_density()
EXPda.plot_sample_density()
ax7.set_title('sampling density [away]')
ax11.set_title('sampling density [away]')


# AWAY, single-start, propagators
PROPs.set_target_axis(ax=ax12)
EXPsh.set_target_axis(ax=ax12)
PROPs.plot_prop_kernels(first_state=state_away_default, cmap=cmap_prop, autoprop_off=autoprop_off, n=1, vmin=0, vmax=vmax_prop)
EXPsh.plot_trajectory(state_seq=[state_away_default], plot_env=False, state_func_env=True)
ax12.set_title('superdiff. propagator [away]')

PROPd.set_target_axis(ax=ax13)
EXPdh.set_target_axis(ax=ax13)
PROPd.plot_prop_kernels(first_state=state_away_default, cmap=cmap_prop, autoprop_off=autoprop_off, n=1, vmin=0, vmax=vmax_prop)
EXPdh.plot_trajectory(state_seq=[state_away_default], plot_env=False, state_func_env=True)
ax13.set_title('diffusive propagator [away]')

# AWAY,multi-start, propagators
PROPs.set_target_axis(ax=ax14)
EXPsh.set_target_axis(ax=ax14)
PROPs.etO[0,:] = prop_density_sa # hack
PROPs.plot_prop_kernels(first_state=0, cmap=cmap_prop, autoprop_off=False, n=1, vmin=0, vmax=vmax_prop)
for start in states_away:
    EXPsh.plot_trajectory(state_seq=[start], plot_env=False, state_func_env=True)
ax14.set_title('superdiff. propagator [away]')

PROPd.set_target_axis(ax=ax15)
EXPdh.set_target_axis(ax=ax15)
PROPd.etO[0,:] = prop_density_da # hack
PROPd.plot_prop_kernels(first_state=0, cmap=cmap_prop, autoprop_off=False, n=1, vmin=0, vmax=vmax_prop)
for start in states_away:
    EXPdh.plot_trajectory(state_seq=[start], plot_env=False, state_func_env=True)
ax15.set_title('diffusive propagator [away]')


x = -0.2; y= 1.2
label_panel(ax0, 'A', x, y)
label_panel(ax1, 'B', x, y)
label_panel(ax2, 'C', x, y)
label_panel(ax3, 'D', x, y)

label_panel(ax4, 'E', x, y)
label_panel(ax5, 'F', x, y)
label_panel(ax6, 'G', x, y)
label_panel(ax7, 'H', x, y)

label_panel(ax8, 'I', x, y)
label_panel(ax9, 'J', x, y)
label_panel(ax10, 'K', x, y)
label_panel(ax11, 'L', x, y)

label_panel(ax12, 'M', x, y)
label_panel(ax13, 'N', x, y)
label_panel(ax14, 'O', x, y)
label_panel(ax15, 'P', x, y)

ax0.set_title('home-events')
ax1.set_title('away-events')
ax2.set_title('sampling density [home]')
ax3.set_title('sampling density [away]')


fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.7, hspace=0.6)
fig.set_size_inches(width, height)

if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
