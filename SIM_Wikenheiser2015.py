#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from environments import CircularTrack
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from visualization import change_seaborn_bar_width

run_lookahead_sim = True
run_tests = False
outdir = os.path.abspath(os.path.join(os.getcwd(), 'figures/Wikenheiser2015'))

color_goals = [(164/255.,171/255.,203/255.), (220/255.,150/255.,150/255.), (64/255.,64/255.,64/255.)]

# SETTINGS - ENVIRONMENT
n_state_circ = 27 # int divisible by 3
start = 4
goal_no = 1

# SETTINGS - GENERATOR
symmetrize = False
jump_rate = 1.

# SETTINGS - PROPAGATOR
sigma = 0.5
tau = 1.
alpha_diff = 1.
alpha_sdiff = 0.5

# SETTINGS - SIMULATOR/LEARNER/EXPLORER
no_dwell = True
n_step = 5
n_samp = 20

# VISUALIZATION
jitter_std = 0.02
traj_width = 0.
state_msize = 40


def wikenheiser2015_lookahead(goal_no, start, alpha, tau, goal_weight=20., goal_absorb=True, backmove=False, opt_policy_weight=0.9, sigma=0.5, n_samp=n_samp, n_step=n_step):
    ENVlh = CircularTrack(n_state=n_state_circ,
                          start=start, goal_no=goal_no,
                          goal_weight=goal_weight,
                          goal_absorb=goal_absorb,
                          backmove=backmove,
                          opt_policy_weight=opt_policy_weight)
    GENlh = Generator(ENV=ENVlh, symmetrize=symmetrize, jump_rate=jump_rate)
    PROPlh = Propagator(GEN=GENlh, sigma=sigma, tau=tau, alpha=alpha)
    EXPlh = Explorer(PROP=PROPlh, rho_init=start, no_dwell=no_dwell)
    EXPlh.sample_sequences(n_samp=n_samp, n_step=n_step)
    EXPlh.traj_width = 0
    EXPlh.start_pos = True
    EXPlh.state_msize = state_msize
    return EXPlh, PROPlh, GENlh, ENVlh

def circulartrack_lookahead(start, alpha, tau, sigma=1, n_samp=n_samp, n_step=n_step):
    ENVlh = CircularTrack(n_state=n_state_circ, start=start, goal_no=None)
    GENlh = Generator(ENV=ENVlh, symmetrize=symmetrize, jump_rate=jump_rate)
    PROPlh = Propagator(GEN=GENlh, sigma=sigma, tau=tau, alpha=alpha)
    EXPlh = Explorer(PROP=PROPlh, rho_init=start, no_dwell=no_dwell)
    EXPlh.sample_sequences(n_samp=n_samp, n_step=n_step)
    EXPlh.traj_width = 0
    EXPlh.start_pos = True
    EXPlh.state_msize = state_msize
    return EXPlh, PROPlh, GENlh, ENVlh


def plot_wikenheiser2015_lookahead(la_df, alpha=alpha_sdiff, tau=tau):
    df = la_df[(la_df.alpha==alpha)&(la_df.tau==tau)]
    sb.set(style="ticks", font_scale=3.5, font={'name':'Arial', 'family':'sans-serif', 'weight':'bold'}, rc={'axes.linewidth':5, 'ytick.major.size':14, 'ytick.major.width':5})
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(12,8))
    ax0 = axes[0]
    sb.barplot(data=df[df.start_type=='init'], x='goal_no', y='look_ahead', estimator=np.mean, palette=color_goals, ax=ax0)
    change_seaborn_bar_width(ax0, 0.5)
    ax0.set_xlabel('')
    ax0.set_xticklabels(['','',''])
    ax0.tick_params(axis='y', which='major', pad=15)
    ax0.tick_params(axis='y', direction='in')
    ax0.tick_params(bottom=False)
    ax0.set_yticks([0,2,4,6,8,10,12])
    ax0.set_ylabel('Look-ahead dist. (a.u.)')
    ax0.set_title('Trajectory initiation', pad=35, size=22)
    ax0.set_ylim([0,12])
    sb.despine(ax=ax0, top=True, right=True)

    ax1 = axes[1]
    sb.barplot(data=df[df.start_type=='arriving'], x='goal_no', y='look_ahead', estimator=np.mean, palette=color_goals, ax=ax1)
    change_seaborn_bar_width(ax1, 0.5)
    ax1.set_xlabel('')
    ax1.set_xticklabels(['','',''])
    ax1.tick_params(axis='y', which='major', pad=15)
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(bottom=False)
    ax1.set_yticks([0,2,4,6,8,10,12])
    ax1.set_ylabel('Look-ahead dist. (a.u.)')
    ax1.set_title('Goal arrival', pad=35, size=22)
    ax1.set_ylim([0,12])
    sb.despine(ax=ax1, top=True, right=True)

    fig.tight_layout()
    return fig
