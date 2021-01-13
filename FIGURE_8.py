#!/usr/bin/python
# -*- coding: utf-8 -*-

# {Suh  J.  Foster  D. J.  Davoudi  H.  Wilson  M. A. \& Tonegawa  S. Impaired hippocampal ripple-associated replay in a mouse model of schizophrenia. \textit{Neuron} \textbf{80}  484-493 (2013).}

# {Karlsson  M. P. \& Frank  L. M. Awake replay of remote experiences in the hippocampus. \textit{Nat Neurosci} \textbf{12}  913-918 (2009).}


import os

import matplotlib.pyplot as plt
import seaborn as sb

from visualization import label_panel, page_width, row_height, color_turb, color_diff, save_figure
from scipy.ndimage.filters import gaussian_filter1d
from environments import LinearTrack, OpenBox
from generators import Generator
from propagators import Propagator
from explorers import Explorer


figdir = os.path.join(os.getcwd(), 'figures')
fname_Karlsson2009 = 'Karlsson2009_fig4_panel.jpg'
fname_Suh2013 = 'Suh2013_KO.png'
save_output = True


# SETTINGS - ENVIRONMENT
scale_track = 50
scale_OB = 40
start = 'default'

# SETTINGS - GENERATOR
symmetrize = True
jump_rate = 1.

# SETTINGS - PROPAGATOR
nu = None # 0.001
sigma = 1. # 1.
alpha_base = 1.
tau_base = 1.
tau = 20. # 20.
alpha_diff = 1. # 1.
alpha_turb = 2 # 2
spec_noise = 0. # 0.

# SETTINGS - EXPLORER
n_samp = 100 # 100
n_step = 150 # 100
mass = 1. # 1.
no_dwell = True # True
diagnostics = False


# SETTINGS - VISUALIZATION
state_msize = 4
kwargs = {'jitter_std':0.02, 'state_msize':state_msize, 'state_lw':0.5, 'traj_width':0.5, 'traj_format':'-o', 'color_time':True, 'cmap_samp':'husl', 'cmap_traj':plt.cm.cool}



# SAMPLE CROSS-CORRELOGRAMS
ENV = LinearTrack(scale=scale_track)
if start == 'default':
    ENV.start = ENV.start_center
else:
    ENV.start = start

GEN = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)
# generate trajectories under diffusion (alpha) and turbulence (+ spectral noise)
PROPt = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_turb, spec_noise=spec_noise)
PROPd = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_diff, spec_noise=0.)
PROPd.plot_prop_kernels(n=6)
EXPt = Explorer(PROP=PROPt, rho_init=ENV.start, mass=mass, no_dwell=no_dwell)
EXPd = Explorer(PROP=PROPd, rho_init=ENV.start, mass=mass, no_dwell=no_dwell)
EXPt.set_viz_scheme(**kwargs)
EXPd.set_viz_scheme(**kwargs)
EXPt.sample_sequences(n_samp=n_samp, n_step=n_step)
EXPd.sample_sequences(n_samp=n_samp, n_step=n_step)
EXPt.plot_trajectory()
EXPd.plot_trajectory()
EXPt.crosscorr_rho(smooth_sigma=0.)
EXPd.crosscorr_rho(smooth_sigma=0.5)


# SAMPLE TRAJECTORIES
ENV_OB = OpenBox(scale=scale_OB)
GEN_OB = Generator(ENV=ENV_OB, symmetrize=symmetrize, jump_rate=jump_rate)
PROPt_OB = Propagator(GEN=GEN_OB, sigma=sigma, tau=tau, alpha=alpha_turb, spec_noise=spec_noise)
PROPd_OB = Propagator(GEN=GEN_OB, sigma=sigma, tau=tau, alpha=alpha_diff, spec_noise=0.)
EXPt_OB = Explorer(PROP=PROPt_OB, rho_init=ENV_OB.start, mass=mass, no_dwell=True)
EXPd_OB = Explorer(PROP=PROPd_OB, rho_init=ENV_OB.start, mass=mass, no_dwell=True)
EXPt_OB.sample_sequences(n_samp=1, n_step=10)
EXPd_OB.sample_sequences(n_samp=1, n_step=10)


# %% FIGURES
width = page_width*2/3.
height = row_height*3
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(width, height))

# log spectral power
PROPt.compute_power_spectrum(alpha_base=alpha_base, tau_base=tau_base)
PROPd.compute_power_spectrum(alpha_base=alpha_base, tau_base=tau_base)

PROPd.plot_relative_power_spectrum(target_ax=axes[0][0], plot_base=True, legend=False)
PROPt.plot_relative_power_spectrum(target_ax=axes[0][0], plot_base=False, legend=False)
axes[0][0].set_xlim([-4,0])
axes[0][0].set_ylim([0,0.1])
axes[0][0].legend()
current_handles, current_labels = axes[0][0].get_legend_handles_labels()
current_labels[0] = r'baseline $[\alpha=1, \tau=1]$'
current_labels[1] = r'diffusion $[\alpha=1, \tau=20]$'
current_labels[2] = r'turbulence $[\alpha=2,\tau=20]$'
axes[0][0].legend(current_handles, current_labels, bbox_to_anchor=(0.12,1), loc='upper left')

# plot exemplar trajectories in open box for clarity
EXPt_OB.set_viz_scheme(color_time=False, color_traj=color_turb, state_msize=state_msize)
EXPd_OB.set_viz_scheme(color_time=False, color_traj=color_diff, state_msize=state_msize)
EXPt_OB.set_target_axis(axes[0][1])
EXPd_OB.set_target_axis(axes[0][1])
EXPt_OB.plot_trajectory()
EXPd_OB.plot_trajectory()
axes[0][1].set_title('sampled sequences', pad=3)

# compute cross-correlations
EXPs = [EXPd, EXPt]
titles = ['cross-correlogram: diffusion', 'cross-correlogram: turbulence']
cbar = [False,False]
min_dist = [1,3] # cut-off artifacts
for ix, ax in enumerate(axes[1][0:2]):
    EXP = EXPs[ix]
    dist_ix = range(min_dist[ix],n_step//5)
    ccorrs = EXP.ccorrs.values[dist_ix,:]
    ccorrs_smooth = gaussian_filter1d(ccorrs, sigma=1, axis=1)
    sb.heatmap(data=ccorrs_smooth, cmap='jet', xticklabels=EXP.time_disp, yticklabels=EXP.dist[dist_ix], linewidths=0, cbar=cbar[ix], ax=ax, rasterized=True)
    ax.invert_yaxis()
    ax.set_ylabel('Place Field Distance')
    ax.set_xlabel('Relative Spike Timing')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[ix])
    sb.despine(ax=ax, left=True)

# add data panels
ax = axes[2][0]
ax.imshow(plt.imread(os.path.join(figdir,fname_Karlsson2009))) # Panel
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Karlsson & Frank, Nat. Neurosci. (2009)', pad=8)
sb.despine(ax=ax, left=True, bottom=True)
ax.set_aspect('auto')
ax = axes[2][1]
ax.imshow(plt.imread(os.path.join(figdir,fname_Suh2013)))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Suh et al., Neuron (2013)', pad=10)
sb.despine(ax=ax, left=True, bottom=True)


x = -0.25
y = 1.2
label_panel(ax=axes[0][0], label='A', x=x, y=y+0.1)
label_panel(ax=axes[0][1], label='B', x=x-0.12, y=y+0.1)
label_panel(ax=axes[1][0], label='C', x=x, y=y)
label_panel(ax=axes[1][1], label='D', x=x+0.1, y=y)
label_panel(ax=axes[2][0], label='E', x=x, y=y+.07)
label_panel(ax=axes[2][1], label='F', x=x+0.1, y=y+.1)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.5, hspace=0.6)
fig.set_size_inches(width, height)
fname_base = 'FIGURE_8'

if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
