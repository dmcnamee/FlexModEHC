#!/usr/bin/python
# -*- coding: utf-8 -*-

# %% {Pfeiffer  B. E. \& Foster  D. J. Autoassociative dynamics in the generation of sequences of hippocampal place cells. \textit{Science} \textbf{349}  180-184 (2015).}

import os

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import visualization as vis

from visualization import save_figure, color_diff, color_superdiff, label_panel, page_width, row_height
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.colors import to_rgba
from environments import OpenBox, LinearTrack
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from scipy.stats import cumfreq


from visualization import save_figure, color_diff, color_superdiff, label_panel, label_panels, page_width, row_height



figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures/'))
fname_base = 'FIGURE_2'
fname_Pfeiffer_panel = 'Pfeiffer_CDF_panel.png'
save_output = True


# && LINEAR TRACK ------------------------------------------------------ ##########


# SETTINGS - ENVIRONMENT
n_state = 10
# goal = n_state-1
goal = None

# SETTINGS - GENERATOR
forward = True
symmetrize = False
jump_rate = 1.

# SETTINGS - PROPAGATOR
alpha_base = 1.
tau_base = 1.
alpha_shift = 0.5
tau_shift = 0.5
beta = 1.
egrad = 1.

# SETTINGS - SIMULATOR/LEARNER/EXPLORER
rho_init = None
mass = 0.
no_dwell = True
diagnostics = True
n_step = 5
n_samp = 1

# VISUALIZATION
jitter_std = 0.03
traj_width = 0.

# %% SIMULATIONS
n_x = 100
alphas = [alpha_shift, alpha_base]
taus = [tau_shift, tau_base]
# alpha_tau_combos = list(product(alphas,taus))
alpha_tau_combos = [(alpha_base,tau_base), (alpha_base, tau_shift), (alpha_shift, tau_shift)]
n_alpha = len(alphas)
n_tau = len(taus)


ENV = LinearTrack(scale=n_state, goal=goal)
GEN = Generator(ENV=LinearTrack(scale=n_state, goal=goal), forward=forward, symmetrize=symmetrize, jump_rate=jump_rate)

evals = GEN.evals
x_state = np.linspace(0,1,n_state)
x = np.linspace(0,1,n_x)
comps = range(1,n_state+1)
comps_rev = [c for c in reversed(comps)]
# generate pandas dataaset
iterables_prop = [alphas, taus, x]
iterables_eval = [alphas, taus, comps]
iterables_comp = [comps, x]
ix_prop = pd.MultiIndex.from_product(iterables_prop, names=['alpha','tau','x'])
ix_eval = pd.MultiIndex.from_product(iterables_eval, names=['alpha','tau','comp'])
ix_comp = pd.MultiIndex.from_product(iterables_comp, names=['comp','x'])
df_prop = pd.DataFrame(index=ix_prop, columns=['prob','cdf'], dtype='float')
df_spec = pd.DataFrame(index=ix_eval, columns=['eval', 'evec_no', 'gain', 'log_gain'], dtype='float')
df_comp = pd.DataFrame(index=ix_comp, columns=['evec_no', 'phi','log_phi'], dtype='float')
for alpha,tau in alpha_tau_combos:
        PROP = Propagator(GEN=GEN, tau=tau, alpha=alpha, beta=beta)
        y_state = PROP.etO[0,:].real
        y = interp1d(x_state, y_state, kind='cubic')
        df_prop.loc[pd.IndexSlice[alpha,tau,:],'prob'] = y(x)
        df_prop.loc[pd.IndexSlice[alpha,tau,:],'cdf'] = 1.- y(x)
        df_spec.loc[pd.IndexSlice[alpha,tau,:],'eval'] = evals
        df_spec.loc[pd.IndexSlice[alpha,tau,:],'evec_no'] = comps
        df_spec.loc[pd.IndexSlice[alpha,tau,:],'gain'] = np.diag(PROP.wetD)
        df_spec.loc[pd.IndexSlice[alpha,tau,:],'log_gain'] = np.log(np.diag(PROP.wetD))

for comp in comps:
    y_state = PROP.U[:,comp-1].real
    phi_func = interp1d(x_state, y_state, kind='cubic')
    phi = phi_func(x)
    log_phi = np.log(phi-phi.min()+0.01)
    df_comp.loc[pd.IndexSlice[comp,:],'evec_no'] = comp
    df_comp.loc[pd.IndexSlice[comp,:],'phi'] = phi
    df_comp.loc[pd.IndexSlice[comp,:],'log_phi'] = log_phi


# flatten dataframes
df_prop = pd.DataFrame(df_prop.to_records()).astype('float')
df_spec = pd.DataFrame(df_spec.to_records()).astype('float')
df_comp = pd.DataFrame(df_comp.to_records()).astype('float')
df_prop_all = df_prop.copy()
df_spec_all = df_spec.copy()
df_comp_all = df_comp.copy()
df_prop_base = df_prop[(df_prop.alpha==alpha_base)&(df_prop.tau==tau_base)]
df_spec_base = df_spec[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_base)]

# compute differences over tau and alphas values
# alpha=1/tau=1 as a baseline
df_prop_base = df_prop_base.loc[(df_prop_base.alpha==alpha_base)&(df_prop_base.tau==tau_base)]
df_prop_diff = df_prop.copy()
df_prop_ratio = df_prop.copy()
df_prop_diff['prob'] = df_prop_diff.groupby(['alpha','tau'])['prob'].apply(lambda x: x-df_prop_base.prob.values)
df_prop_ratio['prob'] = df_prop_ratio.groupby(['alpha','tau'])['prob'].apply(lambda x: x/df_prop_base.prob.values)

# gain relative to alpha=alpha_base/tau=tau_base baseline
df_spec_base = df_spec_base.loc[(df_spec_base.alpha==alpha_base)&(df_spec_base.tau==tau_base)]
df_spec_rel = df_spec.copy()
df_spec_rel['log_gain'] = df_spec_rel.groupby(['alpha','tau'])['log_gain'].apply(lambda x: x-df_spec_base.log_gain.values)
df_spec_rel['gain'] = df_spec_rel.groupby(['alpha','tau'])['gain'].apply(lambda x: x/df_spec_base.gain.values)

# gain relative to total absolute gain
df_spec_rel['log_gain'] = df_spec_rel.groupby(['alpha','tau'])['log_gain'].apply(lambda x: x - x.sum())
df_spec_rel['gain'] = df_spec_rel.groupby(['alpha','tau'])['gain'].apply(lambda x: x/x.sum())



# && OPEN BOX ------------------------------------------------------ ##########

# SETTINGS - ENVIRONMENT
scale = 50 # 50

# SETTINGS - GENERATOR
forward = True
symmetrize = False
jump_rate = 1

# SETTINGS - PROPAGATOR
sigma = 1.
tau_diff = 1.
tau_sdiff = 1.
n_tau = 2; taus = np.logspace(0.1,10,n_tau)
alpha_diff = 1.
alpha_sdiff = 0.5
no_dwell = False
no_dwell_pfeiffer = True

# SETTINGS - EXPLORER
n_step = 75
n_samp = 20
target_coverage = 0.5
flight_vision = True


# SAMPLING
ENV = OpenBox(scale=scale)
start_prop = ENV.start_center

GEN = Generator(ENV=ENV, forward=forward, symmetrize=symmetrize, jump_rate=jump_rate)

PROP_diff_base = Propagator(GEN=GEN, sigma=sigma, tau=tau_diff, alpha=alpha_diff)
PROP_sdiff_base = Propagator(GEN=GEN, sigma=sigma, tau=tau_sdiff, alpha=alpha_sdiff)
PROP_diff_taus = [Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_diff) for tau in taus]
PROP_sdiff_taus = [Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha_sdiff) for tau in taus]

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



# sample many more trajectories to roughly match # trajectory events in Pfeiffer & Foster (2015)
n_samp_mult = 25
EXP_diff_base_LARGESCALE = Explorer(PROP=PROP_diff_base, rho_init=start_prop, no_dwell=no_dwell_pfeiffer, label='diffusion')
EXP_sdiff_base_LARGESCALE = Explorer(PROP=PROP_sdiff_base, rho_init=start_prop, no_dwell=no_dwell_pfeiffer, label='superdiffusion')
EXP_diff_base_LARGESCALE.sample_sequences(n_samp=n_samp*n_samp_mult, n_step=n_step)
EXP_sdiff_base_LARGESCALE.sample_sequences(n_samp=n_samp*n_samp_mult, n_step=n_step)
EXP_diff_base_LARGESCALE.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)
EXP_sdiff_base_LARGESCALE.compute_diagnostics(target_coverage=target_coverage, flight_vision=flight_vision)



# %% FIGURE
width = page_width*1.3
height = row_height*3
widths = [1,1,1,1]
heights = [1,1,1]
legend = None
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True, gridspec_kw={'width_ratios':widths, 'height_ratios':heights})
cmap_prop = {alpha_shift:color_superdiff, alpha_base:color_diff} # cm.cool_rq
cmap_spec = {alpha_shift:color_superdiff, alpha_base:color_diff} # cm.cool_r
cmap_samp = 'colorblind'
traj_width = 0.5
sizes = {tau_shift:2., tau_base:1}
axes[0][0].axis('off')
ax0 = axes[0][0]
ax1 = axes[0][1]
ax2 = axes[0][2]
ax3 = axes[0][3]
ax4 = axes[1][0]
ax5 = axes[1][1]
ax6 = axes[1][2]
ax7 = axes[1][3]
ax8 = axes[2][0]
ax9 = axes[2][1]
ax10 = axes[2][2]
ax11 = axes[2][3]


# LINEAR TRACK
# propagation densities
ax = ax3
sb.lineplot(data=df_prop, x='x', y='prob', hue='alpha', size='tau', sizes=sizes, palette=cmap_prop, ax=ax, legend=legend)
# ax0.axhline(y=0)
ax.set_xlim([0,1])
ax.set_ylim([0,None])
ax.set_xlabel('$x$ position')
ax.set_xticks([], [])
ax.set_ylabel(r'$\rho_{\tau,\alpha}$')
ax.set_title('propagation density')
ax.text(x=0.665, y=0.07, s='heavy tail', color=color_superdiff)
ax.text(x=0.5, y=0.45, s='diffusion', color=color_diff)
ax.text(x=0.5, y=0.4, s='superdiffusion', color=color_superdiff)


# spectral components heatmapped
ax = ax1

out = df_comp_all[['comp','x', 'phi']]
n_comp = len(df_comp_all.comp.unique()) - 1
out = out[out.comp!=1]
repeats = 8
mat = np.repeat(a=out.phi.values.reshape((-1,100)), repeats=repeats, axis=0)
mat = np.flipud(mat)
side_borders = np.pad(np.zeros(mat.shape), pad_width=1, mode='constant', constant_values=1.)
cross_borders = np.pad(np.zeros(mat.shape), pad_width=1, mode='constant', constant_values=1.)
cross_gaps = np.ones(cross_borders.shape)

cross_borders[0,:] = 1.
cross_borders[-1,:] = 1.
for n in range(1,n_comp):
    cross_borders[n*repeats-1,:] = 1.
    cross_borders[n*repeats,:] = 1.
    cross_borders[n*repeats+1,:] = 1.
for n in range(1,n_comp):
    cross_gaps[n*repeats,:] = 0.

mat = np.pad(mat, pad_width=1, mode='constant', constant_values=0.)

# delete some rows for symmetry
mat = np.delete(mat, 1, 0)
side_borders = np.delete(side_borders, 1, 0)
cross_borders = np.delete(cross_borders, 1, 0)
cross_gaps = np.delete(cross_gaps, 1, 0)
mat = np.delete(mat, -2, 0)
side_borders = np.delete(side_borders, -2, 0)
cross_borders = np.delete(cross_borders, -2, 0)
cross_gaps = np.delete(cross_gaps, -2, 0)
mat = np.delete(mat, -2, 0)
side_borders = np.delete(side_borders, -2, 0)
cross_borders = np.delete(cross_borders, -2, 0)
cross_gaps = np.delete(cross_gaps, -2, 0)

# mask for overlays
side_borders_m = np.ma.masked_where(side_borders == 0., side_borders)
cross_borders_m = np.ma.masked_where(cross_borders == 0., cross_borders)
cross_gaps_m = np.ma.masked_where(cross_gaps == 1., cross_gaps)

ax.axis('off')
ax.imshow(mat, cmap='jet')
ax.imshow(side_borders_m, cmap='gray')
ax.imshow(cross_borders_m, cmap='gray')
ax.imshow(cross_gaps_m, cmap='gray_r')
# ax.set_ylabel('spatial scale')
# ax.set_xlabel(r'$x$ \text{ position}')

# relative spectral power
ax = ax2
x_index = 'evec_no' # 'eval' or 'evec_no'

sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_base)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_base]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)

sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)

sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_shift)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_superdiff, markeredgecolor=color_superdiff, clip_on=False, zorder=100)

# ax.set_xlim([df_spec_rel[x_index].min(),df_spec_rel[x_index].max()])
ax.set_ylim([0,0.3])
# ax4.invert_xaxis()
if x_index == 'eval':
    ax.set_xlabel('spectral component')
    ax.set_xticks([], minor=False)
    # ax.set_xlabel('eigenvalue $ \lambda$')
    # ax.set_xticks([9,1.1], minor=False)
    # ax.invert_xaxis()
    ax.set_xlim([-4,0])
else:
    ax.set_xlabel('spatial scale', labelpad=-1)
    # ax.set_xlabel('spatial wavelength')
    ax.set_xticks([1.,10.], minor=False)
    ax.set_xticks(df_spec_rel[x_index].unique(), minor=True)
    ax.set_xticklabels(['large', 'small'], minor=False)
    # ax.tick_params(axis='x', which='minor', bottom=True)
    # ax.tick_params(axis='x', which='major', bottom=False)
    ax.invert_xaxis()
# ax.set_xscale('log')
# ax.set_ylabel(r'$r_{\tau,\alpha}(\lambda)$   [normalized]')
ax.set_ylabel(r'$s_{\alpha,\tau}(\lambda) / s_{1,1}(\lambda)$ [normalized]')
ax.set_title('relative power spectrum')



# OPENBOX CONTRIBUTIONS
ax4.set_title('diffusion (single)')
ax5.set_title('superdiffusion (single)')
ax6.set_title('diffusion (multiple)')
ax7.set_title('superdiffusion (multiple)')

EXP_diff_oneshot.cmap_samp = cmap_samp
EXP_diff_oneshot.traj_width = traj_width

EXP_sdiff_oneshot.cmap_samp = cmap_samp
EXP_sdiff_oneshot.traj_width = traj_width

EXP_diff_base.cmap_samp = cmap_samp
EXP_diff_base.traj_width = traj_width

EXP_sdiff_base.cmap_samp = cmap_samp
EXP_sdiff_base.traj_width = traj_width

# EXP_diff_base.set_target_axis(ax=ax4)
# EXP_diff_base.plot_trajectory()
# EXP_sdiff_base.set_target_axis(ax=ax5)
# EXP_sdiff_base.plot_trajectory()
EXP_diff_oneshot.set_target_axis(ax=ax4)
EXP_diff_oneshot.plot_trajectory()
EXP_sdiff_oneshot.set_target_axis(ax=ax5)
EXP_sdiff_oneshot.plot_trajectory()
EXP_diff_base.set_target_axis(ax=ax6)
EXP_diff_base.plot_trajectories()
EXP_sdiff_base.set_target_axis(ax=ax7)
EXP_sdiff_base.plot_trajectories()

EXP_diff_base.set_target_axis(ax=ax9)
EXP_diff_base.plot_coverage(color=color_diff, func_of_time=False)
EXP_sdiff_base.set_target_axis(ax=ax9)
EXP_sdiff_base.plot_coverage(color=color_superdiff, func_of_time=False)
ax9.set_ylabel('fraction of env. visited', labelpad=5)
ax9.set_title('exploration efficiency (multiple)', pad=10)
ax9.set_xlabel('avg. distance traversed')
# ax9.set_xlim([0, EXP_diff_base.traj_cost_mean.max()])
ax9.set_xlim([0, 150])
ax9.set_ylim([0,0.6])
# ax9.legend(loc='upper right')

# mean/sem across sampled sequences
EXP_diff_base.set_target_axis(ax=ax8)
EXP_diff_base.plot_coverage(color=color_diff, func_of_time=False, across_samp=True)
EXP_sdiff_base.set_target_axis(ax=ax8)
EXP_sdiff_base.plot_coverage(color=color_superdiff, func_of_time=False, across_samp=True)
# parallelized coverage across sampled sequences
# EXP_diff_oneshot.set_target_axis(ax=ax8)
# EXP_diff_oneshot.plot_coverage(color=color_diff, func_of_time=False)
# EXP_sdiff_oneshot.set_target_axis(ax=ax8)
# EXP_sdiff_oneshot.plot_coverage(color=color_superdiff, func_of_time=False)
ax8.set_ylabel('fraction of env. visited', labelpad=5)
ax8.set_title('exploration efficiency (single)', pad=10)
ax8.set_xlabel('avg. distance traversed')
# ax8.set_xlim([0, EXP_diff_oneshot.traj_cost_mean.max()])
ax8.set_xlim([0, 150])
ax8.set_ylim([None,0.06])
# ax8.legend(['superdiffusion', 'diffusion'], loc='upper right')
ax8.text(x=8, y=0.055, s='diffusion', color=color_diff)
ax8.text(x=8, y=0.05, s='superdiffusion', color=color_superdiff)


# plt.setp(ax9.get_yticklabels(), visible=False)
# ax9.set_ylabel('visited locations / distance', labelpad=5)
# ax9.set_title('exploration efficiency', pad=10)
# ax9.set_yticks([])
# ax9.set_xlabel('avg. distance traversed')
# ax9.set_xlim([0, EXP_diff_base.traj_cost_mean.max()])
# # ax9.get_legend().remove()
# ax9.legend(loc='lower right')

# CDFs
numbins = 100
cdf_diff = cumfreq(EXP_diff_base_LARGESCALE.jump_length.flatten(), numbins=numbins)
cdf_sdiff = cumfreq(EXP_sdiff_base_LARGESCALE.jump_length.flatten(), numbins=numbins)

y_diff = cdf_diff.cumcount/cdf_diff.cumcount.max()
y_sdiff = cdf_sdiff.cumcount/cdf_sdiff.cumcount.max()

x_diff = np.linspace(0, cdf_diff.binsize*numbins, numbins)
x_sdiff = np.linspace(0, cdf_sdiff.binsize*numbins, numbins)

# roughly rescale spatial scale to match that in Pfeiffer & Foster (2015)
# using sdiff vs empirical data as a reference
x_diff = 200*x_diff/x_sdiff.max()
x_sdiff = 200*x_sdiff/x_sdiff.max()

y_diff_smooth = gaussian_filter1d(y_diff, 4., mode='nearest')
y_sdiff_smooth = gaussian_filter1d(y_sdiff, 1., mode='nearest')

# ax10.plot(x_diff, y_diff, color=color_diff, label='diffusion', clip_on=False)
# ax10.plot(x_sdiff, y_sdiff, color=color_superdiff, label='superdiffusion', clip_on=True)

ax10.plot(x_diff, y_diff_smooth, color=color_diff, label='diffusion', clip_on=False)
ax10.plot(x_sdiff, y_sdiff_smooth, color=color_superdiff, label='superdiffusion', clip_on=True)
ax10.set_xlim([0,50])
ax10.set_ylim([0.,1.])
ax10.set_xticks([0,50])
ax10.set_yticks([0,1])
ax10.set_xlabel('step size (a.u.)')
ax10.set_ylabel('cumulative fraction')
ax10.set_title('model simulation', pad=8)

image = plt.imread(os.path.join(figdir, fname_Pfeiffer_panel))
ax11.imshow(image)
ax11.axis('off')
ax11.set_title('Pfeiffer & Foster, Science (2015)', pad=9)
ax11.text(x=90, y=460, s='empirical data', color='black', fontsize=10)
ax11.text(x=35, y=500, s='even-step prediction', color='red', fontsize=10)
ax11.add_patch(patches.Rectangle((5,420), width=400, height=100, fill=False, ec='black', clip_on=False))

# tweak panels
sb.despine(fig, top=True, right=True)
x = -0.3
y = 1.2
label_panel(ax0, 'A', x, 1.2)
label_panel(ax1, 'B', x, 1.2+0.2)
label_panel(ax2, 'C', x, 1.2)
label_panel(ax3, 'D', x, 1.2)
label_panel(ax4, 'E', x-0.1, 1.3)
label_panel(ax5, 'F', x-0.1, 1.3)
label_panel(ax6, 'G', x-0.1, 1.3)
label_panel(ax7, 'H', x-0.1, 1.3)
label_panel(ax8, 'I', x-.05, 1.3)
label_panel(ax9, 'J', x, 1.3)
label_panel(ax10, 'K', x, 1.3)
label_panel(ax11, 'L', x, 1.3)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.6, hspace=0.6)
fig.set_size_inches(width, height)


if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
