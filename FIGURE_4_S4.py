#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SIM_Wikenheiser2015 import wikenheiser2015_lookahead, plot_wikenheiser2015_lookahead
from visualization import save_figure, change_seaborn_bar_width, font_scale, label_panels, page_width, row_height, color_diff, color_superdiff, interpolate_curve
from matplotlib import colors

plot_lookahead_seqgen = True
plot_allgoal_prop_mat = True
plot_lookahead_dist = True

save_output = True
figdir = os.path.abspath(os.path.join(os.getcwd(), 'figures/'))

# ENVIRONMENT SETTINGS
start = 1
state_ticks = [1,8,17,26]
state_ticklabels = ['X', 'G1', 'G2', 'G3']

# PROPAGATOR SETTINGS
alpha_diff = 1.
alpha_sdiff = 0.5
tau_base = 1.
sigma = 0.5

# SIMULATOR SETTINGS
n_step = 5



# %% ----------- goal-directed propagators: uni-goal propagator graphs and sample trajectories
if plot_lookahead_seqgen:
    goal_absorb = True

    exp_diff_g1, prop_diff_g1, _, env = wikenheiser2015_lookahead(goal_no=1, start=start, alpha=alpha_diff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    exp_diff_g2, prop_diff_g2, _, env = wikenheiser2015_lookahead(goal_no=2, start=start, alpha=alpha_diff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    exp_sdiff_g1, prop_sdiff_g1, _, _ = wikenheiser2015_lookahead(goal_no=1, start=start, alpha=alpha_sdiff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    exp_sdiff_g2, prop_sdiff_g2, _, _ = wikenheiser2015_lookahead(goal_no=2, start=start, alpha=alpha_sdiff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    exp_sdiff_g3, prop_sdiff_g3, _, _ = wikenheiser2015_lookahead(goal_no=3, start=start, alpha=alpha_sdiff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    n_state = env.n_state; states = np.arange(n_state)

    # sequence generation
    exp_diff_g1.sample_sequences(n_samp=1, n_step=n_step)

    # sequence plots
    figsize = (12,12)
    exp_diff_g1.plot_trajectory(samp=0, plot_env=True, state_func_env=False, figsize=figsize)
    if save_output:
        fname_base = 'FIGURE_4_sample_diff_g1'
        save_figure(fig=exp_diff_g1.fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=exp_diff_g1.fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
    exp_sdiff_g1.plot_trajectory(samp=0, plot_env=True, state_func_env=False, figsize=figsize)
    if save_output:
        fname_base = 'FIGURE_4_sample_sdiff_g1'
        save_figure(fig=exp_sdiff_g1.fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=exp_sdiff_g1.fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
    exp_sdiff_g2.plot_trajectory(samp=0, plot_env=True, state_func_env=False, figsize=figsize)
    if save_output:
        fname_base = 'FIGURE_4_sample_sdiff_g2'
        save_figure(fig=exp_sdiff_g2.fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=exp_sdiff_g2.fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')


    # graph propagators
    fig = plt.figure(figsize=(page_width/2.,row_height)); axg = plt.gca()
    P_diff_g1 = prop_diff_g1.etO[start,:]
    P_diff_g2 = prop_diff_g2.etO[start,:]
    P_sdiff_g1 = prop_sdiff_g1.etO[start,:]
    P_sdiff_g2 = prop_sdiff_g2.etO[start,:]
    P_sdiff_g3 = prop_sdiff_g3.etO[start,:]

    states_smooth, P_diff_g1_smooth = interpolate_curve(states, P_diff_g1)
    _, P_diff_g2_smooth = interpolate_curve(states, P_diff_g2)
    _, P_sdiff_g1_smooth = interpolate_curve(states, P_sdiff_g1)
    _, P_sdiff_g2_smooth = interpolate_curve(states, P_sdiff_g2)
    _, P_sdiff_g3_smooth = interpolate_curve(states, P_sdiff_g3)

    axg.plot(states_smooth, P_diff_g1_smooth, label='diffusive [Goal 1, 2, or 3]', color=color_diff)
    axg.plot(states_smooth, P_sdiff_g1_smooth, label='superdiffusive [Goal 1]', color=color_superdiff)
    axg.plot(states_smooth, P_sdiff_g2_smooth, label='superdiffusive [Goal 2]', linestyle='--', color=color_superdiff)
    axg.plot(states_smooth, P_sdiff_g3_smooth, label='superdiffusive [Goal 3]', linestyle=':', color=color_superdiff)
    axg.set_xlabel('circular track position')
    axg.set_ylabel('propagator')
    axg.set_xticks(state_ticks)
    axg.set_xticklabels(state_ticklabels)
    axg.set_ylim([0,None])
    axg.legend()
    if save_output:
        fname_base = 'FIGURE_4_prop'
        save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')


# %% ----------- multi-goal propagator matrices
if plot_allgoal_prop_mat:
    goal_absorb = False
    goal_no = 'all'

    fig_mat, axes_mat = plt.subplots(nrows=1, ncols=3, figsize=(1.3*page_width,row_height), sharex=True, sharey=True)

    ax0 = axes_mat[0]; ax1 = axes_mat[1]; ax2 = axes_mat[2]
    exp_sdiff_all, prop_sdiff_all, _, _ = wikenheiser2015_lookahead(goal_no=goal_no, start=start, alpha=alpha_sdiff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)
    exp_diff_all, prop_diff_all, _, _ = wikenheiser2015_lookahead(goal_no=goal_no, start=start, alpha=alpha_diff, tau=tau_base, sigma=sigma, n_samp=1, goal_absorb=goal_absorb)

    prop_sdiff_all.set_target_axis(ax=ax0)
    prop_sdiff_all.plot_prop_kernels_matrix(off_diagonal=True, offdiag_cbar=True, interpolation='spline36', state_ticks=state_ticks, state_ticklabels=state_ticklabels)
    ax0.set_title('superdiffusive propagator')
    ax0.axhline(y=start, color='white', linestyle='-', linewidth=1)

    prop_diff_all.set_target_axis(ax=ax1)
    prop_diff_all.plot_prop_kernels_matrix(off_diagonal=True, offdiag_cbar=True, interpolation='spline36', state_ticks=state_ticks, state_ticklabels=state_ticklabels)
    ax1.set_title('diffusive propagator')
    ax1.axhline(y=start, color='white', linestyle='-', linewidth=1)

    # propagator difference matrix plot (adapted from Propagator.plot_prop_kernels_matrix)
    diff = prop_sdiff_all.etO - prop_diff_all.etO
    # vmin = None; vmax = None
    vmin = -0.075; vmax = 0.075
    colornorm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
    imdiff = ax2.imshow(diff, origin='upper', cmap=plt.cm.RdBu, interpolation='spline36', norm=colornorm)
    ax2.set_ylabel('current position')
    ax2.set_xlabel('future position')
    ax2.set_title(r'propagation difference $\rho_{\alpha=0.5}- \rho_{\alpha=1}$', pad=10)
    ax2.grid(color='gray', linestyle='-', linewidth=0.5)
    ax2.axhline(y=start, color='black', linestyle='-', linewidth=1)
    ax2.plot(list(reversed(ax2.get_xlim())), ax2.get_ylim(), ls="-", c=".3")
    plt.gcf().colorbar(mappable=imdiff, shrink=1., ax=ax2)
    # goal-specific labeling
    ax2.set_xticks(state_ticks)
    ax2.set_yticks(state_ticks)
    ax2.set_xticklabels(state_ticklabels)
    ax2.set_yticklabels(state_ticklabels)
    ax0.set_frame_on(False)
    ax1.set_frame_on(False)
    ax2.set_frame_on(False)
    plt.tight_layout()

    if save_output:
        fname_base = 'FIGURE_S4'
        save_figure(fig=fig_mat, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=fig_mat, figdir=figdir, fname_base=fname_base, file_ext='.pdf')




# %% LOOK-AHEAD DISTANCE SIMULATION PLOTS
if plot_lookahead_dist:
    # Data were aligned to trajectory initiation, divided by how far the rat would run and examined over the initial limb of each trajectory (shaded region).
    # The look-ahead distance of each theta cycle was the distance between the rat’s location and the average of place field centers of cells active in the final quarter of that theta cycle, weighted by the number of spikes each cell fired.
    # we model it has the distance to the furthest place activation.

    # ENVIRONMENT/TASK SETTINGS
    # track is 80cm in diameter therefore circumference is 80pi
    diameter = 80
    track_length = diameter*np.pi
    state_interval = track_length/500 # scale to match Wikenheiser2015 scale
    start_init = 1
    start_arriving = [4,13,22] # states for look-ahead computations on goal arrival analyses (i.e. states just past centers of greyed segments in Fig 4a)
    goal_nos = [1,2,3]
    start_types = ['init', 'arriving']
    n_samp_lh = 50

    # SET UP DATAFRAME
    n_ix = len(goal_nos)*2*n_samp_lh
    la_df = pd.DataFrame(index=pd.Index(range(n_ix)), columns=['tau', 'alpha', 'goal_no', 'start', 'start_type', 'samp', 'final_state', 'look_ahead'])

    # SAMPLE LOOK-AHEADS
    ix = 0
    for goal_no in goal_nos:
        for i,startlh in enumerate([start_init, start_arriving[goal_no-1]]):
            start_type = start_types[i]
            EXPlh,_,_,ENVlh = wikenheiser2015_lookahead(goal_no=goal_no, start=startlh, alpha=alpha_sdiff, tau=tau_base, sigma=sigma, n_samp=n_samp_lh)
            for samp in range(n_samp_lh):
                final_state = EXPlh._retrieve_state(samp=samp, step=int(n_step), coords=False)
                look_ahead = ENVlh.distance(startlh, final_state, interval_size=state_interval)
                la_df.loc[ix,'tau'] = tau_base
                la_df.loc[ix,'alpha'] = alpha_sdiff
                la_df.loc[ix,'goal_no'] = goal_no
                la_df.loc[ix,'start'] = startlh
                la_df.loc[ix,'start_type'] = start_type
                la_df.loc[ix,'samp'] = samp
                la_df.loc[ix,'final_state'] = final_state
                la_df.loc[ix,'look_ahead'] = look_ahead
                ix += 1

    fig_lh = plot_wikenheiser2015_lookahead(la_df, alpha=alpha_sdiff, tau=tau_base)
    if save_output:
        fname_base = 'FIGURE_4_lookahead_dist'
        save_figure(fig=fig_lh, figdir=figdir, fname_base=fname_base, file_ext='.png')
        save_figure(fig=fig_lh, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
