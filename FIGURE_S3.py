#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import config

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import visualization as vis

from matplotlib.colors import to_rgba
from environments import RoomWorld
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from scipy.stats import cumfreq

# SETTINGS - ENVIRONMENT
scale = 25


figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures'))
fname_base = 'FIGURE_S3'
save_output = True

# SETTINGS - GENERATOR
symmetrize = False
jump_rate = 15

# SETTINGS - PROPAGATOR
tau = 1. # 1, 5, 10 tested
tau_diff = tau
tau_sdiff = tau
alpha_diff = 1.
alpha_sdiff = 0.5
no_dwell = False

# SETTINGS - EXPLORER
n_step = 75
n_samp = 20
target_coverage = 0.5
flight_vision = True


# SAMPLING
ENV = RoomWorld(scale=scale)
start_prop = ENV.start_center_TL

GEN = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)

PROP_diff_base = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROP_sdiff_base = Propagator(GEN=GEN, tau=tau_sdiff, alpha=alpha_sdiff)

# parallel explorer
EXP_diff_base = Explorer(PROP=PROP_diff_base, rho_init=start_prop, no_dwell=no_dwell, label='diffusion')
EXP_sdiff_base = Explorer(PROP=PROP_sdiff_base, rho_init=start_prop, no_dwell=no_dwell, label='superdiffusion')
EXP_diff_base.sample_sequences(n_samp=n_samp, n_step=n_step)
EXP_sdiff_base.sample_sequences(n_samp=n_samp, n_step=n_step)
EXP_diff_base.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_sdiff_base.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)

# serial explorer
EXP_diff_oneshot = Explorer(PROP=PROP_diff_base, rho_init=start_prop, no_dwell=no_dwell, label='diffusion')
EXP_sdiff_oneshot = Explorer(PROP=PROP_sdiff_base, rho_init=start_prop, no_dwell=no_dwell, label='superdiffusion')
EXP_diff_oneshot.sample_sequences(n_samp=1, n_step=n_step)
EXP_sdiff_oneshot.sample_sequences(n_samp=1, n_step=n_step)
EXP_diff_oneshot.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_sdiff_oneshot.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)







# %% FIGURES
from visualization import save_figure, color_diff, color_superdiff, label_panel, page_width, row_height
cmap_samp = 'colorblind'
traj_width = 0.5
width = page_width*0.5
height = row_height*3
widths = [1,1]
heights = [1,1,1]
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True, gridspec_kw={'width_ratios':widths, 'height_ratios':heights})

ax0 = axes[0][0]
ax1 = axes[0][1]
ax2 = axes[1][0]
ax3 = axes[1][1]
ax4 = axes[2][0]
ax5 = axes[2][1]

EXP_diff_base.ENV.env_lw = 0.5
EXP_sdiff_base.ENV.env_lw = 0.5

ax0.set_title('diffusion (single)')
ax1.set_title('diffusion (multiple)')
ax2.set_title('superdiffusion (single)')
ax3.set_title('superdiffusion (multiple)')

EXP_diff_base.cmap_samp = cmap_samp
EXP_sdiff_base.cmap_samp = cmap_samp
EXP_diff_base.traj_width = traj_width
EXP_sdiff_base.traj_width = traj_width
EXP_diff_base.set_target_axis(ax=ax0)
EXP_diff_base.plot_trajectory()
EXP_diff_base.set_target_axis(ax=ax1)
EXP_diff_base.plot_trajectories()
EXP_sdiff_base.set_target_axis(ax=ax2)
EXP_sdiff_base.plot_trajectory()
EXP_sdiff_base.set_target_axis(ax=ax3)
EXP_sdiff_base.plot_trajectories()

EXP_diff_base.set_target_axis(ax=ax5)
EXP_diff_base.plot_coverage(color=color_diff, func_of_time=False)
EXP_sdiff_base.set_target_axis(ax=ax5)
EXP_sdiff_base.plot_coverage(color=color_superdiff, func_of_time=False)
ax5.set_ylabel('fraction of env. visited', labelpad=5)
ax5.set_title('exploration efficiency (multiple)', pad=10)
ax5.set_xlabel('avg. distance traversed')
ax5.set_xlim([0, 300])
ax5.set_ylim([0,0.8])

# mean/sem across sampled sequences
EXP_diff_base.set_target_axis(ax=ax4)
EXP_diff_base.plot_coverage(color=color_diff, func_of_time=False, across_samp=True)
EXP_sdiff_base.set_target_axis(ax=ax4)
EXP_sdiff_base.plot_coverage(color=color_superdiff, func_of_time=False, across_samp=True)
ax4.set_ylabel('fraction of env. visited', labelpad=5)
ax4.set_title('exploration efficiency (single)', pad=10)
ax4.set_xlabel('avg. distance traversed')
ax4.set_xlim([0, 300])
ax4.set_ylim([0,0.12])
ax4.text(x=20, y=0.1, s='diffusion', color=color_diff)
ax4.text(x=20, y=0.09, s='superdiffusion', color=color_superdiff)


x = -0.35; y = 1.2
label_panel(ax0, 'A', x, y)
label_panel(ax1, 'B', x, y)
label_panel(ax2, 'C', x, y)
label_panel(ax3, 'D', x, y)
label_panel(ax4, 'E', x, y)
label_panel(ax5, 'F', x, y)
fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.7, hspace=0.6)
fig.set_size_inches(width, height)

if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
