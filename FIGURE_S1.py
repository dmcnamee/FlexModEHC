#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import config

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from matplotlib import cm
from scipy.stats import norm
from itertools import product

from environments import LinearTrack
from generators import Generator
from propagators import Propagator
from simulators import Simulator
from explorers import Explorer
from learners import Learner


from visualization import save_figure, color_diff, color_superdiff, label_panel, label_panels, page_width, row_height
plt.style.use(['FlexModEHC.mplrc'])
fname_base = 'FIGURE_S1'
save_output = True


figdir = os.path.join(os.getcwd(), 'figures')


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
alpha_tau_combos = list(product(alphas,taus))
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






# %% FIGURE
width = page_width
height = row_height*2
widths = [1,1,1]
heights = [1,1]
legend = None
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True, gridspec_kw={'width_ratios':widths, 'height_ratios':heights})
cmap_prop = {alpha_shift:color_superdiff, alpha_base:color_diff} # cm.cool_rq
cmap_spec = {alpha_shift:color_superdiff, alpha_base:color_diff} # cm.cool_r
sizes = {tau_shift:2., tau_base:1}
ax0 = axes[0][0]
ax1 = axes[0][1]
ax2 = axes[0][2]
ax3 = axes[1][0]
ax4 = axes[1][1]
ax5 = axes[1][2]


# propagation densities
ax = ax0
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


# relative propagation densities (subtraction)
ax = ax1
sb.lineplot(data=df_prop_diff, x='x', y='prob', hue='alpha', size='tau', sizes=sizes, palette=cmap_prop, ax=ax, legend=legend)
ax.set_xlim([0,1])
# ax1.axhline(y=0, linestyle='--')
ax.set_xlabel('$x$ position')
ax.set_xticks([], [])
ax.set_ylabel(r'$\rho_{\tau,\alpha}-\rho_{1,1}$')
ax.set_title('relative propagation density')


# displacement plots (displays linear vs non-linear rescaling)
ax = ax2
for alpha in alphas:
    for tau in taus:
        df_prop.loc[(df_prop.alpha==alpha)&(df_prop.tau==tau),'prob_base'] = df_prop_base.prob.values
sb.lineplot(data=df_prop, x='prob_base', y='prob', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend)

ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)
ax.set_ylim([10**-5,0.55])
ax.set_xticks(ax.get_yticks(minor=True), minor=True)
ax.set_xticks(ax.get_yticks(minor=False), minor=False)
ax.set_ylim([10**-5,0.55])
ax.set_xlim([10**-5,0.55])
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel(r'$\rho_{1,1}$ at position $x$')
ax.set_ylabel(r'$\rho_{\tau,\alpha}$ at position $x$')
# ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('propagation distortion')
ax.text(x=0.00008, y=0.005, s='linear', color=color_diff)
ax.text(x=0.00045, y=0.12, s='non-linear', color=color_superdiff)


# spectral components
ax = ax3
evec_no_low = 3
evec_no_high = 7
df_comp = df_comp[df_comp.evec_no.isin([evec_no_low,evec_no_high])]
sb.lineplot(data=df_comp, x='x', y='phi', style='evec_no', color='k', ax=ax)
ax.set_xlim([0,1])
ax.set_ylim([-0.5,1.8])
ax.set_xlabel(r'$x$ position')
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.set_xlabel(r'$x$ position')
ax.set_ylabel(r'$\phi(x)$')
ax.set_title('spectral components')
L = ax.legend(loc='upper right', frameon=True, title=None)
handles, labels = ax.get_legend_handles_labels()
L = ax.legend(handles=handles, labels=labels, frameon=True, title=r'$\lambda_{k} = $ e-value (scale)')
L.get_texts()[0].set_text(r'$\lambda_{%i}=%.2f$ (large)'%(evec_no_low,evals[evec_no_low]))
L.get_texts()[1].set_text(r'$\lambda_{%i}=%.2f$ (small)'%(evec_no_high,evals[evec_no_high]))
L._legend_box.align = "right"
plt.setp(L.get_title(),fontsize=10)

# log spectral power
ax = ax4
x_index = 'eval'
sb.lineplot(data=df_spec[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_base)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_base]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)

sb.lineplot(data=df_spec[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)

sb.lineplot(data=df_spec[(df_spec.alpha==alpha_shift)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_superdiff, markeredgecolor=color_superdiff, clip_on=False, zorder=100)
ax.set_xlim([df_spec[x_index].min(),df_spec[x_index].max()])

sb.lineplot(data=df_spec[(df_spec.alpha==alpha_shift)&(df_spec.tau==tau_base)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax, legend=legend, marker='o', markersize=sizes[tau_base]*2, markeredgewidth=0., markerfacecolor=color_superdiff, markeredgecolor=color_superdiff, clip_on=False, zorder=100)
ax.set_xlim([df_spec[x_index].min(),df_spec[x_index].max()])

ax.set_yscale('log')
if x_index == 'eval':
    ax.set_xlabel(r'eigenvalue $\lambda$')
    ax.set_xlim([-4,0])
    ax.set_ylim([None, 1])
else:
    ax.set_xlabel('spatial wavelength')
ax.set_ylabel(r'$s_{\tau,\alpha}(\lambda)$   [log scale]')
ax.set_title('power spectrum')

# relative spectral power
ax = ax5
x_index = 'eval'

sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_base)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax5, legend=legend, marker='o', markersize=sizes[tau_base]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)
sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_base)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax5, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_diff, markeredgecolor=color_diff, clip_on=False, zorder=100)
sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_shift)&(df_spec.tau==tau_shift)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax5, legend=legend, marker='o', markersize=sizes[tau_shift]*2, markeredgewidth=0., markerfacecolor=color_superdiff, markeredgecolor=color_superdiff, clip_on=False, zorder=100)
sb.lineplot(data=df_spec_rel[(df_spec.alpha==alpha_shift)&(df_spec.tau==tau_base)], x=x_index, y='gain', hue='alpha', size='tau', sizes=sizes, palette=cmap_spec, ax=ax5, legend=legend, marker='o', markersize=sizes[tau_base]*2, markeredgewidth=0., markerfacecolor=color_superdiff, markeredgecolor=color_superdiff, clip_on=False, zorder=100)

ax5.set_ylim([0,0.3])
if x_index == 'eval':
    ax5.set_xlabel('eigenvalue $ \lambda$')
    ax5.set_xlim([-4,0])
else:
    ax5.set_xlabel('spatial wavelength')
ax5.set_ylabel(r'$s_{\alpha,\tau}(\lambda) / s_{1,1}(\lambda)$ [normalized]')
ax5.set_title('relative power spectrum')


sb.despine(fig, top=True, right=True)
# label_panels(axes)
x = -0.3
y = 1.2
label_panel(axes[0][0], 'A', x, y)
label_panel(axes[0][1], 'B', x, y)
label_panel(axes[0][2], 'C', x, y)
label_panel(axes[1][0], 'D', x, y)
label_panel(axes[1][1], 'E', x, y)
label_panel(axes[1][2], 'F', x, y)

fig.subplots_adjust(left=.01, bottom=.1, right=.99, top=.9, wspace=0.6, hspace=0.6)
fig.set_size_inches(width, height)
if save_output:
        save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
