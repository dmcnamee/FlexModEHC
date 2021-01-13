#!/usr/bin/python
# -*- coding: utf-8 -*-


# {Stella  F.  Baracskay  P.  O'Neill  J. \& Csicsvari  J. Hippocampal Reactivation of Random Trajectories Resembling Brownian Diffusion. \textit{Neuron} (2019).}


import os
import config

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import visualization as vis

from matplotlib.colors import to_rgba
from environments import OpenBox
from generators import Generator
from propagators import Propagator
from explorers import Explorer
from scipy.io import loadmat
from visualization import save_figure, color_diff, color_superdiff
from scipy.ndimage import gaussian_filter1d


save_output = True

figdir = os.path.abspath(os.path.join(os.getcwd(), "figures/"))
datadir = os.path.abspath(os.path.join(os.getcwd(), "data/StellaNeuron2019"))
file_behave = "Stella2019_Fig4_mu0.71.csv"
file_csv = "Stella2019_MSD_stats.csv"

run_data_comparison = True
include_behave = True
run_Stella2019_prop_kernels = True

plt.style.use(["FlexModEHC.mplrc"])
capsize = 2

def convert_mu_to_alpha(mu):
    return 1 / (2.0 * mu)


def convert_alpha_to_mu(alpha):
    return 1 / (2.0 * alpha)

# SETTINGS
scale = 17
diameter = 120  # cm
area = diameter * np.pi

tau = 1.0
sigma = 10.0
sigma_behave = 4.0


Ks = (sigma_behave ** 2) * np.linspace(0.12, 1.1, 5)  # relevant to behave analysis
K_avg = np.mean(Ks)
t_steps_diff = np.arange(1, 9, 1)
t_steps_superdiff = np.arange(1, 11, 1)
t_steps = np.arange(1, 9, 1)
mus_sleep = [0.53, 0.53, 0.51, 0.52, 0.45]  # from Fig 3A
alphas_sleep = [convert_mu_to_alpha(m) for m in mus_sleep]
mus_awake = [0.54, 0.62, 0.65, 0.64, 0.68]  # from WakeSWR_Exponents.fig
alphas_awake = [convert_mu_to_alpha(m) for m in mus_awake]
mus_behave = [
    0.68,
    0.72,
    0.71,
    0.69,
    0.67,
]  # from Fig 4A, SEMS extracted and loaded below
behave_offsets_deltatime1 = [3.15, 6.13, 9.13, 11.87, 14.51]
alphas_behave = [convert_mu_to_alpha(m) for m in mus_behave]

alpha_wake_avg = np.mean(alphas_awake)
# alpha_sleep_avg = np.mean(alphas_sleep)
alpha_sleep_avg = 1.0  # fix to diffusion (close as makes no difference)
# alpha_behave = np.mean(alphas_behave)
behave_curve_no = 2  # which curve from Stella2019, Fig 3 to plot (plotting 0.71 curve, median example)
alpha_behave = alphas_behave[behave_curve_no]

if run_data_comparison:
    # LOAD DATA
    data_behave = pd.read_csv(os.path.join(datadir, file_behave), index_col=0)
    data_behave = data_behave[0:8]
    df_data = pd.read_csv(os.path.join(datadir, file_csv))

    # SIMULATIONS
    ENV = OpenBox(scale=scale)
    state_size = area / ENV.n_state
    start_prop = ENV.start_center  # None, ENV.start
    GEN = Generator(ENV=ENV, symmetrize=True)

    # PLOT
    markersize = 4
    # mu_sleep_mean/sem
    data_md_sleep_mean = (
        df_data[
            (df_data.state == "sleep")
            & (df_data.stat == "mean")
            & (df_data.type == "actual")
        ]
        .groupby("time_disp")
        .spatial_disp.mean()
    )
    data_md_sleep_sem = (
        df_data[
            (df_data.state == "sleep")
            & (df_data.stat == "sem")
            & (df_data.type == "actual")
        ]
        .groupby("time_disp")
        .spatial_disp.mean()
    )
    K_sleep = sigma * data_md_sleep_mean[1]
    PROP_sleep = Propagator(GEN=GEN, tau=tau, alpha=alpha_sleep_avg)
    sim_msd_sleep_mean = PROP_sleep.msd(t_steps, K=K_sleep)
    sim_md_sleep_mean = PROP_sleep.mean_displacement(t_steps, K=K_sleep)

    # mu_awake_mean/sem
    data_md_wake_mean = (
        df_data[
            (df_data.state == "awake")
            & (df_data.stat == "mean")
            & (df_data.type == "actual")
        ]
        .groupby("time_disp")
        .spatial_disp.mean()
    )
    data_md_wake_sem = (
        df_data[
            (df_data.state == "awake")
            & (df_data.stat == "sem")
            & (df_data.type == "actual")
        ]
        .groupby("time_disp")
        .spatial_disp.mean()
    )
    K_wake = sigma * data_md_wake_mean[1]
    PROP_wake = Propagator(GEN=GEN, tau=tau, alpha=alpha_wake_avg)
    sim_msd_awake_mean = PROP_wake.msd(t_steps, K=K_wake)
    sim_md_awake_mean = PROP_wake.mean_displacement(t_steps, K=K_wake)

    if include_behave:
        K_behave = sigma * behave_offsets_deltatime1[behave_curve_no]
        PROP_behave = Propagator(GEN=GEN, tau=tau, alpha=alpha_behave)
        sim_msd_behave_mean = PROP_behave.msd(t_steps, K=K_behave)
        sim_md_behave_mean = PROP_behave.mean_displacement(t_steps, K=K_behave)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=False,
        sharey=False,
        constrained_layout=True,
        figsize=(vis.page_width / 1.5, vis.row_height),
    )
    # MSDs
    if include_behave:
        ax.plot(
            t_steps,
            sim_md_behave_mean,
            linestyle="dashdot",
            color="blue",
            label=r"superdiffusion $\left[\alpha=%.1f\right]$" % alpha_behave,
            clip_on=False,
        )
    ax.plot(
        t_steps,
        sim_md_awake_mean,
        linestyle="-",
        color="blue",
        label=r"superdiffusion $\left[\alpha=%.1f\right]$" % alpha_wake_avg,
        clip_on=False,
    )
    ax.plot(
        t_steps,
        sim_md_sleep_mean,
        linestyle="-",
        color="red",
        label=r"diffusion $\left[\alpha=%.1f\right]$" % alpha_sleep_avg,
        clip_on=False,
    )

    if include_behave:
        data_behave.plot(
            ax=ax,
            x="x",
            y="y",
            yerr="sem",
            color="blue",
            linestyle="",
            marker="s",
            markersize=markersize,
            label=r"behavior $[\Delta t\approx 620$ms$]$",
            clip_on=False,
            capsize=capsize,
        )

    data_md_wake_mean.plot(
        ax=ax,
        yerr=data_md_wake_sem,
        color="blue",
        linestyle="",
        marker="o",
        markersize=markersize,
        label=r"wake SWRs $[\Delta t\approx 8$ms$]$",
        clip_on=False,
        capsize=capsize,
    )
    data_md_sleep_mean.plot(
        ax=ax,
        yerr=data_md_sleep_sem,
        color="red",
        linestyle="",
        marker="o",
        markersize=markersize,
        label=r"sleep SWRs $[\Delta t\approx 8$ms$]$",
        clip_on=False,
        capsize=capsize,
    )
    ax.tick_params(which="both", left=True, bottom=True)
    ax.set_xlabel(r"Time-step Interval ($\Delta t$)")
    ax.set_ylabel("Mean displacement (cm)")
    # log scale
    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)
    ax.set_xlim([None, 10 ** 1])
    ax.set_ylim([None, 10 ** 2])
    ax.set_yticks([10 ** 1, 10 ** 2])
    leg = ax.legend(bbox_to_anchor=(1.05, 0.9), frameon=False)
    fig.tight_layout()
    if save_output:
        fig.savefig(os.path.join(figdir, "FIGURE_S7_msd_wake_sleep_logscale.png"), dpi=300)
        fig.savefig(os.path.join(figdir, "FIGURE_S7_msd_wake_sleep_logscale.pdf"), dpi=300)

    # linear scale
    ax.set_xscale("linear", basex=10)
    ax.set_yscale("linear", basey=10)
    # ax.set_aspect("equal")
    ax.set_xlim([None, 8.0])
    if include_behave:
        ax.set_ylim([None, 50])
    else:
        ax.set_ylim([None, 50])
    ax.set_xticklabels([0, 2, 4, 6, 8])
    ax.set_yticks([10 ** 1, 50])
    for x in ax.get_children():
        x.set_clip_on(False)
    fig.tight_layout()
    if save_output:
        fig.savefig(os.path.join(figdir, "FIGURE_6_msd_wake_sleep_linscale.png"), dpi=300)
        fig.savefig(os.path.join(figdir, "FIGURE_6_msd_wake_sleep_linscale.pdf"), dpi=300)


if run_Stella2019_prop_kernels:
    ENV = OpenBox(scale=scale)
    start_prop = ENV.start_center  # None, ENV.start
    GEN = Generator(ENV=ENV, symmetrize=True)

    # plots of local diffusion density (Stella, Fig. 6E)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        sharex=True,
        sharey=True,
        figsize=(vis.page_width, 2 * vis.row_height),
    )
    # DIFF
    alpha = 1.0
    taus = 10 * np.flip(np.arange(1, 5))
    for i, tau in enumerate(taus):
        ax = axes[0][i]
        PROP = Propagator(GEN=GEN, tau=tau, alpha=alpha)
        PROP.set_target_axis(ax=ax)
        PROP.plot_prop_kernels(
            n=1,
            first_state=start_prop,
            wrap_col=4,
            cmap=plt.cm.hot,
            midpoint_norm=False,
            autoprop_off=True,
            cbar=False,
        )
        ax.set_title(r"$\Delta t=%i$" % (i + 1))

    # SUPERDIFF
    alpha = 0.5
    taus = 10 * np.flip(np.arange(1, 5))
    for i, tau in enumerate(taus):
        ax = axes[1][i]
        PROP = Propagator(GEN=GEN, tau=tau, alpha=alpha)
        PROP.set_target_axis(ax=ax)
        PROP.plot_prop_kernels(
            n=1,
            first_state=start_prop,
            wrap_col=4,
            cmap=plt.cm.hot,
            midpoint_norm=False,
            autoprop_off=True,
            cbar=False,
        )
        ax.set_title(r"$\Delta t=%i$" % (i + 1))

    fig.tight_layout()
    if save_output:
        plt.savefig(os.path.join(figdir, "FIGURE_6_prop_kernels.png"), dpi=300)
        plt.savefig(os.path.join(figdir, "FIGURE_6_prop_kernels.pdf"), dpi=300)
