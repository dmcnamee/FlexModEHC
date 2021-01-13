#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt

from environments import RoomWorld
from generators import Generator, modify_jump_rates, adjmat2generator
from propagators import Propagator
from simulators import Simulator
from explorers import Explorer
from learners import Learner
from visualization import save_figure, page_width, row_height, label_panels

# NOTE use state_plot_env = False (to replicate black-background propagator density plots)


run_3x3 = True
run_panels = True
save_output = True

fname_base = "FIGURE_S2"
figdir = os.path.join(os.getcwd(), "figures")

# SETTINGS - ENVIRONMENT
scale_highres = 30
scale_lowres = 15

# SETTINGS - GENERATOR
kernels = None
n = 5
forward = True
symmetrize = True
jump_rate = 1.0
norm_scale = False

# SETTINGS - PROPAGATOR
sigma = 1.0
taus = [10, 0.1, 0.05]
ts = [0, 1, 2]
n_taus = len(taus)
alphas = [0.5, 1.0]

# SETTINGS - SIMULATOR/LEARNER/EXPLORER
no_dwell = False
n_samp = 1

# VISUALIZATION
state_msize = 5


# %% eigenvectors panels
ENV = RoomWorld(start=0, scale=scale_highres)
kernels = [3, 8, 23, 35, 9*8, 9*12+1, 16*12, 16*12+1, 17*12+1, 19*12-4]
ncols = len(kernels)
GEN = Generator(ENV=ENV, forward=forward, symmetrize=symmetrize, jump_rate=jump_rate)

GEN.plot_real_eigenvectors(kernels=kernels, wrap_col=ncols, title=False, norm_scale=norm_scale)
if save_output:
    GEN.fig.savefig(
        os.path.join(figdir, fname_base + "_evecs.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


# %% tau x t and sample panels
ENV = RoomWorld(start=0, scale=scale_lowres)
start = ENV.n_state // 4 - 10
n_step = 5
for alpha in alphas:
    width = page_width
    height = row_height * 3
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(width, height))
    for ix_tau, tau in enumerate(taus):
        GEN = Generator(
            ENV=ENV, forward=forward, symmetrize=symmetrize, jump_rate=jump_rate
        )
        PROP = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha)
        SIM = Simulator(PROP=PROP, rho_init=start, no_dwell=no_dwell)
        SIM.sample_sequences(n_samp=1, n_step=n_step)
        samps = SIM._retrieve_state(coords=False).squeeze()
        SIM.set_target_axis(ax=axes[ix_tau][3])
        SIM.color_time = True
        SIM.state_msize = state_msize
        SIM.plot_trajectory()
        for ix_t, t in enumerate(ts):
            PROP.set_target_axis(ax=axes[ix_tau][ix_t])
            PROP.plot_prop_kernels(
                n=1, first_state=int(samps[ix_t]), wrap_col=1, autoprop_off=False
            )
            axes[ix_tau][ix_t].set_title("")
    fig.subplots_adjust(
        left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0.5, hspace=0.6
    )
    fig.set_size_inches(width, height)
    if save_output:
        save_figure(
            fig=fig,
            figdir=figdir,
            fname_base=fname_base + "_TauxT_alpha%.1f" % alpha,
            file_ext=".png",
        )
        save_figure(
            fig=fig,
            figdir=figdir,
            fname_base=fname_base + "_TauxT_alpha%.1f" % alpha,
            file_ext=".pdf",
        )


# %% 3x3 figure
if run_3x3:
    ENV = RoomWorld(start=0, scale=scale_lowres)
    n_step = 30
    start = ENV.n_state // 4 - 1
    width = page_width
    height = row_height * 3
    for alpha in alphas:
        fig, axes = plt.subplots(
            nrows=3, ncols=n_taus, figsize=(width, height), sharey=False
        )

        # GENERATORS
        GEN = Generator(
            ENV=ENV, forward=forward, symmetrize=symmetrize, jump_rate=jump_rate
        )
        start = 8
        for ix, tau in enumerate(taus):
            GEN.set_target_axis(ax=axes[0][ix])
            GEN.plot_real_eigenvectors(
                start=start,
                n=1,
                wrap_col=1,
                title=False,
                norm_scale=norm_scale,
            )
            start += 1

        for ix, tau in enumerate(taus):
            # PROPAGATORS
            PROP = Propagator(GEN=GEN, sigma=sigma, tau=tau, alpha=alpha)
            PROP.set_target_axis(ax=axes[1][ix])
            PROP.plot_prop_kernels(
                n=1, first_state=start, wrap_col=1, autoprop_off=False, cbar=False
            )
            axes[1][ix].set_title(r"$\tau=%.2f$" % tau)

            # SIMULATORS
            SIM = Simulator(PROP=PROP, rho_init=start, no_dwell=no_dwell)
            SIM.sample_sequences(n_samp=n_samp, n_step=n_step)
            SIM.color_time = True
            SIM.state_msize = state_msize
            SIM.set_target_axis(ax=axes[2][ix])
            SIM.plot_trajectory()
        label_panels(axes, x=-0.15, y=1.2)

        fig.subplots_adjust(
            left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0.3, hspace=0.5
        )
        fig.set_size_inches(width, height)
        if save_output:
            save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".png")
            save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext=".pdf")
