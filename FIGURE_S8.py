#!/usr/bin/python
# -*- coding: utf-8 -*-

# pattern formation dynamics of non-negative generator factorization
# https://github.com/ganguli-lab/grid-pattern-formation

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as to
import torch.nn.functional as F
import seaborn as sb

from environments import OpenBox
from generators import Generator
from rl_graph import successor_rep
from place_cells import PredictivePlaceCells
from grid_cells import plot_ratemaps
from visualization import cmap_grid_code, rgb, MidpointNormalize, color_diff, color_superdiff, save_figure
from tqdm import tqdm
from matplotlib import cm

from pattern_formation import ortho_grid_pattern_formation, grid_pattern_formation, Gaussian, DoG, diff_gen
from utils import ensure_dir, generate_run_ID_PATFORM
from generators import adjmat2generator, generator2stochmat, adjmat2stochmat
from scipy.ndimage import gaussian_filter

figdir = os.path.abspath(os.path.join(os.getcwd(), "figures"))
fname_base = "FIGURE_S8"
save_output = True


class Options:
    pass

options = Options()
options.COV = 'GEN'
options.res = 32*4
options.Nx = options.res**2
options.Np = 512
options.Ng = 64
options.T = 20000
options.lr = 8e-3
options.r = 100
options.place_cell_rf = 0.12
options.surround_scale = 2
options.MODEL_type = 'GEN_PATFORM'
options.activation = 'relu'
options.DoG = False
options.gauss_norm = False
options.periodic = False
options.norm_cov = False
options.gauss_norm = False
options.box_width = 2.2
options.box_height = 2.2



# run grid pattern formation
Q = diff_gen(options=options, sigma=0.5, surround_scale=1.5)
options.Q = to.as_tensor(Q)

place_cells = PredictivePlaceCells(options=options)

place_cells.compute_covariance(predmap=True)
place_cells.Cmean = gaussian_filter(input=place_cells.Cmean, sigma=4., truncate=6., order=0, mode='wrap', cval=0.0)
place_cells.fft_covariance(predmap=True)

# rescale/smooth in fourier space
place_cells.Ctilde /= place_cells.Ctilde.max()
place_cells.Ctilde *= options.Ng

# run pattern formation
Gmaps = ortho_grid_pattern_formation(place_cells, options, activation='relu')

# plot resulting maps
fig = plot_ratemaps(np.flip(Gmaps), options.Ng)
if save_output:
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.png')
    save_figure(fig=fig, figdir=figdir, fname_base=fname_base, file_ext='.pdf')
