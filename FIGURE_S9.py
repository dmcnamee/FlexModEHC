#!/usr/bin/python
# -*- coding: utf-8 -*-

# {Stensola  H. et al. The entorhinal grid map is discretized. \textit{Nature} \textbf{492}  72-78 (2012).}


import os
import config

import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import to_rgba
from environments import OpenBox
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from scipy.stats import cumfreq
from scipy.linalg import expm
from scipy.special import softmax, kl_div
from scipy.interpolate import interp1d
from sklearn.preprocessing import minmax_scale
from scipy.stats import norm
from visualization import color_diff, color_superdiff
from copy import deepcopy
from scipy.optimize import minimize, Bounds
from functools import partial


save_output = True

# SETTINGS - EXPERIMENT
# [Stensola2012] hypothesis regarding areal doubling
scale_ratio = np.sqrt(2)
peak_weights = [1,1,1,1]
n_peaks = len(peak_weights)
scale_ratio_var = 0.03
pop_ratio = 0.75

# SETTINGS - ENVIRONMENT
scale = 15

figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures'))
fname_base = 'FIGURE_S9'
save_output = True

# SETTINGS - GENERATOR
symmetrize = True
jump_rate = 1

# SETTINGS - PROPAGATOR
tau_diff = 16.
tau_sdiff = 8.

alpha_diff = 1.
alpha_sdiff = 0.5

# SETTINGS - EXPLORER
n_step_sdiff = int(75*(15/50))
n_step_diff = int(75*(15/50)) + 10
n_samp = 20 # 20
n_step_fig = 10
n_samp_fig = 10
no_dwell = True
flight_vision = True
target_coverage = 0.5
across_samp = True



# SAMPLING
ENV = OpenBox(scale=scale)
start = ENV.start_center

GEN = Generator(ENV=ENV, symmetrize=symmetrize, jump_rate=jump_rate)
GEN_sr = deepcopy(GEN)
EVEC = GEN.EVEC.copy()
n_state = GEN.n_state



# %% sub-sampling scale-ratio spectral components
# create cell histogram as a function of scale
scale_space_res = n_state
spec_comps = np.flip(np.arange(n_state))
scales = np.flip(GEN.evals)
scale_min = scales.min()
scale_max = scales.max()
scale_space = np.linspace(scale_min, scale_max, scale_space_res)
scale_range = scale_max - scale_min

# organize cluster means according to scale-ratio
spec_clusters = np.ones((n_peaks,))
for i in range(1,n_peaks):
    spec_clusters[i:] *= scale_ratio
spec_clusters = minmax_scale(spec_clusters, feature_range=(scale_min + scale_ratio_var*scale_range, scale_max - scale_ratio_var*scale_range))

pop_pdf = np.zeros((scale_space_res,))
for i,mean in enumerate(spec_clusters):
    pop_pdf += peak_weights[i]*norm.pdf(scale_space, loc=mean, scale=scale_ratio_var*scale_range)
pop_pdf /= pop_pdf.sum()


# sample spectral components from pop_pdf without replacement
pop_size = int(pop_ratio*n_state)
spec_comps_sr = np.random.choice(spec_comps, size=pop_size, replace=False, p=pop_pdf)
spec_comps_sr = np.unique(spec_comps_sr)[::-1]
spec_comps_discard = [i for i in range(n_state) if i not in spec_comps_sr]
G = EVEC.copy()
G = np.delete(G, spec_comps_discard, 1)
scales_sr = scales[spec_comps_sr]

# invert for readout
Ginv = la.pinv(G)
D = Ginv@GEN.Q@G






# %% construct propagators and sample
thresh = 0.0

def prop_thresh(prop, thresh=0.):
    prop[np.where(prop<thresh)] = 0
    return prop

def prop_bias(prop, bias=0.):
    prop -= bias
    return prop

def prop_thresh_maxnorm(prop, thresh=0.):
    prop[np.where(prop<thresh*prop.max())] = 0
    return prop

def prop_norm(prop):
    prop /= prop.sum(axis=1, keepdims=True)
    return prop

def prop_sr(tau=1., alpha=1., spectrum=None, thresh=0., norm=True):
    if spectrum is None:
        prop = G@expm(-(1/tau)*(np.abs(D)**alpha))@Ginv
    else:
        prop = G@np.diag(spectrum)@Ginv
    prop = prop_thresh(prop, thresh=thresh)
    if norm:
        prop = prop_norm(prop)
    return prop

def loss_autoprop(tau, alpha, target, state):
    etO_sr = prop_sr(tau, alpha=alpha)
    return np.square(etO_sr[state,state] - target)

def loss_kl_prop(spectrum, prop_target):
    prop_spec = prop_sr(spectrum=spectrum, thresh=thresh, norm=True)
    err = kl_div(prop_spec.flatten(), prop_target.flatten())
    err[np.where(np.isinf(err))] = kl_div(0.1,0.01)
    return err.sum()

def loss_kl_prop_joint(params, prop_diff_target, prop_sdiff_target):
    raise ValueError('untested')
    n_spec = int((params.size - 1)//2)

    spec_diff = params[1:n_spec+1]
    spec_sdiff = params[n_spec+1:]
    thresh = params[0]
    prop_diff_spec = prop_sr(spectrum=spec_diff, thresh=thresh, norm=True)
    prop_sdiff_spec = prop_sr(spectrum=spec_sdiff, thresh=thresh, norm=True)
    err_diff = kl_div(prop_diff_spec.flatten(), prop_diff_target.flatten())
    err_sdiff = kl_div(prop_sdiff_spec.flatten(), prop_sdiff_target.flatten())
    err_diff[np.where(np.isinf(err_diff))] = kl_div(0.1,0.01)
    err_sdiff[np.where(np.isinf(err_sdiff))] = kl_div(0.1,0.01)
    return err_diff.sum() + err_sdiff.sum()

PROP_diff = Propagator(GEN=GEN, tau=tau_diff, alpha=alpha_diff)
PROP_sdiff = Propagator(GEN=GEN, tau=tau_sdiff, alpha=alpha_sdiff)
PROP_diff_sr = Propagator(GEN=GEN_sr, tau=tau_diff, alpha=alpha_diff)
PROP_sdiff_sr = Propagator(GEN=GEN_sr, tau=tau_sdiff, alpha=alpha_sdiff)
# make temp copies
etO_diff = PROP_diff.etO.copy()
etO_sdiff = PROP_sdiff.etO.copy()


# spectrum-optimization via KL-divergence
tau_diff_sr = tau_diff
alpha_diff_sr = alpha_diff
tau_sdiff_sr = tau_sdiff
alpha_sdiff_sr = alpha_sdiff

diff_sr_obj = partial(loss_kl_prop, prop_target=etO_diff)
sdiff_sr_obj = partial(loss_kl_prop, prop_target=etO_sdiff)

spec_init_diff = np.exp(-(1/tau_diff_sr)*(np.abs(np.diag(D))**alpha_diff_sr))
spec_init_sdiff = np.exp(-(1/tau_sdiff_sr)*(np.abs(np.diag(D))**alpha_sdiff_sr))

bounds = Bounds(lb=0, ub=np.inf)
options = {'maxiter': 1000, 'disp': True}
res_diff = minimize(fun=diff_sr_obj, x0=spec_init_diff, bounds=bounds, options=options)
spec_sr_diff = res_diff.x
res_sdiff = minimize(fun=sdiff_sr_obj, x0=spec_init_sdiff, bounds=bounds, options=options)
spec_sr_sdiff = res_sdiff.x


# %% set up scale-ratio propagators
PROP_diff_sr.etO = prop_sr(spectrum=spec_sr_diff)
PROP_sdiff_sr.etO = prop_sr(spectrum=spec_sr_sdiff)

# bias
bias = 0.0
PROP_diff_sr.etO = prop_bias(PROP_diff_sr.etO, bias=bias)
PROP_sdiff_sr.etO = prop_bias(PROP_sdiff_sr.etO, bias=bias)

# threshold
thresh = 0.0
PROP_diff_sr.etO = prop_thresh(PROP_diff_sr.etO, thresh=thresh)
PROP_sdiff_sr.etO = prop_thresh(PROP_sdiff_sr.etO, thresh=thresh)

# normalize
PROP_diff_sr.etO = prop_norm(PROP_diff_sr.etO)
PROP_sdiff_sr.etO = prop_norm(PROP_sdiff_sr.etO)

PROP_diff.plot_prop_kernels(n=5, first_state=start, wrap_col=5, autoprop_off=False, cbar=False)
PROP_diff_sr.plot_prop_kernels(n=5, first_state=start, wrap_col=5, autoprop_off=False, cbar=False)
PROP_sdiff.plot_prop_kernels(n=5, first_state=start, wrap_col=5, autoprop_off=False, cbar=False)
PROP_sdiff_sr.plot_prop_kernels(n=5, first_state=start, wrap_col=5, autoprop_off=False, cbar=False)



# %% explorers - sample trajectories
EXP_diff = Explorer(PROP=PROP_diff, rho_init=start, no_dwell=no_dwell, label='diffusion [complete]')
EXP_diff_sr = Explorer(PROP=PROP_diff_sr, rho_init=start, no_dwell=no_dwell, label='diffusion [scale ratio]')
EXP_sdiff = Explorer(PROP=PROP_sdiff, rho_init=start, no_dwell=no_dwell, label='superdiffusion [complete]')
EXP_sdiff_sr = Explorer(PROP=PROP_sdiff_sr, rho_init=start, no_dwell=no_dwell, label='superdiffusion [scale ratio]')

EXP_diff.sample_sequences(n_samp=n_samp, n_step=n_step_diff)
EXP_diff_sr.sample_sequences(n_samp=n_samp, n_step=n_step_diff)
EXP_sdiff.sample_sequences(n_samp=n_samp, n_step=n_step_sdiff)
EXP_sdiff_sr.sample_sequences(n_samp=n_samp, n_step=n_step_sdiff)

EXP_diff.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_diff_sr.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_sdiff.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_sdiff_sr.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)

# sample fewer steps/samps for visual clarity
EXP_diff_fig = Explorer(PROP=PROP_diff, rho_init=start, no_dwell=no_dwell, label='diffusion [complete]')
EXP_diff_sr_fig = Explorer(PROP=PROP_diff_sr, rho_init=start, no_dwell=no_dwell, label='diffusion [scale ratio]')
EXP_sdiff_fig = Explorer(PROP=PROP_sdiff, rho_init=start, no_dwell=no_dwell, label='superdiffusion [complete]')
EXP_sdiff_sr_fig = Explorer(PROP=PROP_sdiff_sr, rho_init=start, no_dwell=no_dwell, label='superdiffusion [scale ratio]')
EXP_diff_fig.sample_sequences(n_samp=n_samp_fig, n_step=n_step_fig)
EXP_diff_sr_fig.sample_sequences(n_samp=n_samp_fig, n_step=n_step_fig)
EXP_sdiff_fig.sample_sequences(n_samp=n_samp_fig, n_step=n_step_fig)
EXP_sdiff_sr_fig.sample_sequences(n_samp=n_samp_fig, n_step=n_step_fig)



# %% FIGURES
from visualization import save_figure, color_diff, color_superdiff, label_panel, label_panels, page_width, row_height
plt.rcParams["axes.axisbelow"] = False
cmap_samp = 'colorblind'
traj_width = 0.5
width = page_width
height = row_height*3
widths = [1,1,1,1]
heights = [0.75,1,1]
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True, gridspec_kw={'width_ratios':widths, 'height_ratios':heights})

EXP_diff.ENV.env_lw = 0.5
EXP_sdiff.ENV.env_lw = 0.5
EXP_diff_sr.ENV.env_lw = 0.5
EXP_sdiff_sr.ENV.env_lw = 0.5

ax0 = axes[1][0]
ax1 = axes[1][1]
ax2 = axes[2][0]
ax3 = axes[2][1]
ax4 = axes[1][2]
ax5 = axes[1][3]
ax6 = axes[2][2]
ax7 = axes[2][3]
ax8 = axes[0][0]
ax9 = axes[0][1]
ax10 = axes[0][2]
ax11 = axes[0][3]

# plot coverage stats
EXP_diff.set_target_axis(ax=ax10)
EXP_sdiff.set_target_axis(ax=ax10)
EXP_diff_sr.set_target_axis(ax=ax11)
EXP_sdiff_sr.set_target_axis(ax=ax11)
EXP_diff.plot_coverage(color=color_diff, func_of_time=False, across_samp=across_samp)
EXP_sdiff.plot_coverage(color=color_superdiff, func_of_time=False, across_samp=across_samp)
EXP_diff_sr.plot_coverage(color=color_diff, func_of_time=False, across_samp=across_samp)
EXP_sdiff_sr.plot_coverage(color=color_superdiff, func_of_time=False, across_samp=across_samp)
ax10.set_ylabel('fraction of env. visited', labelpad=0)
ax10.set_title('dense population', pad=10)
ax10.set_xlabel('avg. distance traversed')
ax10.set_ylim([0,0.25])
ax10.set_yticks([0,0.05,0.1,0.15,0.2,0.25])
ax10.text(x=4, y=0.18, s='diffusion', color=color_diff)
ax10.text(x=4, y=0.22, s='superdiffusion', color=color_superdiff)
ax11.set_ylim([0,0.25])
ax11.set_yticks([0,0.05,0.1,0.15,0.2,0.25])
ax11.set_ylabel('fraction of env. visited', labelpad=0)
ax11.set_title('scale-ratio population', pad=10)
ax11.set_xlabel('avg. distance traversed')
ax10.set_xlim([0,75])
ax11.set_xlim([0,75])
ax10.set_xticks([0,75])
ax11.set_xticks([0,75])


# plot individual samples
EXP_diff_fig.cmap_samp = cmap_samp
EXP_diff_sr_fig.cmap_samp = cmap_samp
EXP_diff_fig.traj_width = traj_width
EXP_diff_sr_fig.traj_width = traj_width
EXP_diff_fig.set_target_axis(ax=ax0)
EXP_diff_fig.plot_trajectory()
EXP_diff_fig.set_target_axis(ax=ax1)
EXP_diff_fig.plot_trajectories()
EXP_diff_sr_fig.set_target_axis(ax=ax2)
EXP_diff_sr_fig.plot_trajectory()
EXP_diff_sr_fig.set_target_axis(ax=ax3)
EXP_diff_sr_fig.plot_trajectories()

EXP_sdiff.cmap_samp = cmap_samp
EXP_sdiff_sr.cmap_samp = cmap_samp
EXP_sdiff.traj_width = traj_width
EXP_sdiff_sr.traj_width = traj_width
EXP_sdiff.set_target_axis(ax=ax4)
EXP_sdiff.plot_trajectory()
EXP_sdiff.set_target_axis(ax=ax5)
EXP_sdiff.plot_trajectories()
EXP_sdiff_sr.set_target_axis(ax=ax6)
EXP_sdiff_sr.plot_trajectory()
EXP_sdiff_sr.set_target_axis(ax=ax7)
EXP_sdiff_sr.plot_trajectories()

ax0.set_title('diffusion (single)', pad=-10)
ax1.set_title('diffusion (multiple)', pad=-10)
ax2.set_title('diffusion (single)', pad=-10)
ax3.set_title('diffusion (multiple)', pad=-10)
ax4.set_title('superdiffusion (single)', pad=-10)
ax5.set_title('superdiffusion (multiple)', pad=-10)
ax6.set_title('superdiffusion (single)', pad=-10)
ax7.set_title('superdiffusion (multiple)', pad=-10)


# plot spectral component density function
for mean in spec_clusters:
    ax8.axvline(x=mean, ls='--', color='grey', lw=1, zorder=99)
ax8.axvline(scale_min, color='red', lw=2, zorder=99, clip_on=False)
ax8.axvline(scale_max, color='red', lw=2, zorder=99, clip_on=False)

ax8.plot(scale_space, pop_pdf, color='k', zorder=98)
ax8.fill_between(y1=pop_pdf, x=scale_space, color='k', zorder=98)
ax8.set_ylim([0,None])
ax8.set_ylabel('density', labelpad=0)
ax8.set_xlabel('spatial scale', labelpad=5)
ax8.set_xticks([scale_space.min(),scale_space.max()], minor=False)
ax8.set_xticklabels(['smaller', 'larger'], minor=False)
ax8.set_xlim([None,scale_space.max()])

# plot sampled spectral components
ax9.plot(spec_comps, scales, color='k')
ax9.plot(spec_comps_sr, scales[::-1][spec_comps_sr], 'x', color='k', markersize=2.5, label='sampled', clip_on=False)
ax9.set_ylabel('generator eigenvalue', labelpad=0)
ax9.invert_xaxis()
ax9.legend(loc='upper left', bbox_to_anchor=(-0.1,1.1), handletextpad=0.005)
ax9.set_xlabel('spectral component (scale)', labelpad=5)
ax9.set_xticks([spec_comps_sr.min(),spec_comps_sr.max()], minor=False)
ax9.set_xticklabels(['larger', 'smaller'], minor=False)
ax9.set_xlim([None,spec_comps_sr.min()])
ax9.set_ylim([None,0.])

x = -0.35; y = 1.3
label_panels(axes, x, y)
fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.9, hspace=0.9)
fig.set_size_inches(width, height)


if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
