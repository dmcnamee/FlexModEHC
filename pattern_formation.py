# -*- coding: utf-8 -*-

# The code in this file is adapted from https://github.com/ganguli-lab/grid-pattern-formation

# {Sorscher  B.  Mel  G.  Ganguli  S. \& Ocko  S. A unified theory for the origin of grid cells through the lens of pattern formation. \textit{Advances in Neural Information Processing Systems} (2019).}

import matplotlib.pyplot as plt
import numpy as np
import torch as to
import torch.nn.functional as F

from grid_cells import plot_ratemaps
from utils import generate_run_ID_PATFORM
from functools import partial
from tqdm import tqdm

from place_cells import PlaceCells



run_experiments = False




def convolve_with_C(g, Ctilde, res):
    '''
    Convolves the input g with the kernel C
    '''
    gtilde = np.fft.fft2(g, [res, res])
    gconv = np.real(np.fft.ifft2(gtilde*Ctilde))
    gconv = np.roll(np.roll(gconv, res//2+1, axis=1), res//2+1, axis=2)

    return gconv




def grid_pattern_formation(place_cells, options, activation='relu'):
    """ non-orthogonalized single-cell pattern formation, numpy version """
    if hasattr(options, 'T'):
        T = options.T
    else:
        T = 5000
    if hasattr(options, 'lr'):
        lr = options.lr
    else:
        lr = 8e-3
    if hasattr(options, 'r'):
        r = options.r
    else:
        r = 1
    Ng = options.Ng
    res = options.res
    G = np.random.randn(Ng,res,res) * 1e-8
    # C = place_cells.compute_covariance()
    Ctilde = place_cells.fft_covariance() # Fourier transform

    if symmetry is 'break':
        activation = lambda x: np.maximum(x,0)
        sym_string = 'SYMMETRY-BREAKING'
    elif symmetry is 'preserve':
        activation = np.tanh
        sym_string = 'SYMMETRY-PRESERVING'
    else:
        raise ValueError('mis-specification')

    for i in tqdm(range(T), desc='PATTERN FORMATION (NON-ORTHOGONAL, %s)'%sym_string):
        G +=lr*(-G + convolve_with_C(G, Ctilde, res))
        G = activation(G)

        if np.isnan(G).any():
            print('NaNs detected!')
            return Glast
        else:
            Glast = G.copy()
    return G


#
# def ortho_grid_pattern_formation(place_cells, options, activation='relu'):
#     """ orthogonalized multi-cell pattern formation, numpy implementation """
#     if hasattr(options, 'T'):
#         T = options.T
#     else:
#         T = 5000
#     if hasattr(options, 'lr'):
#         lr = options.lr
#     else:
#         lr = 8e-3
#     if hasattr(options, 'r'):
#         r = options.r
#     else:
#         r = 30
#     Ng = options.Ng
#     res = options.res
#     G = np.random.randn(Ng,res,res) * 1e-8
#     # C = place_cells.compute_covariance()
#     Ctilde = place_cells.fft_covariance() # Fourier transform
#
#     if activation == 'relu':
#         activation = lambda x: np.maximum(r*x,0)
#         sym_string = 'SYMMETRY-BREAKING'
#     elif activation == 'tanh':
#         activation = np.tanh
#         sym_string = 'SYMMETRY-PRESERVING'
#     else:
#         raise ValueError('mis-specification')
#
#     for i in tqdm(range(T), desc='PATTERN FORMATION (ORTHOGONAL, %s)'%sym_string):
#         H = convolve_with_C(G, Ctilde, res)
#         Hr = H.reshape([Ng, -1])
#         Gr = G.reshape([Ng, -1])
#         oja = Gr.T.dot(np.tril(Gr.dot(Hr.T))).T.reshape([Ng,res,res])
#         G += lr * (H - oja + activation(G))
#
#         if np.isnan(G).any():
#             raise ValueError('NaNs detected!')
#     return G



def ortho_grid_pattern_formation(place_cells, options, max_norm=False, activation='relu'):
    """ orthogonalized multi-cell pattern formation, torch implementation """
    if hasattr(options, 'T'):
        T = options.T
    else:
        T = 5000
    if hasattr(options, 'lr'):
        lr = options.lr
    else:
        lr = 8e-3
    if hasattr(options, 'r'):
        r = options.r
    else:
        r = 30
    Ng = options.Ng
    res = options.res
    G = to.as_tensor(np.random.randn(Ng,res,res) * 1e-8).double()
    if not hasattr(place_cells, 'Ctilde'):
        Ctilde = to.as_tensor(place_cells.fft_covariance()).double() # Fourier transform
    else:
        Ctilde = to.as_tensor(place_cells.Ctilde)

    if activation == 'relu':
        activation = lambda x: to.relu(r*x)
        sym_string = 'SYMMETRY-BREAKING'
    elif activation == 'relu-sm':
        activation = lambda x: F.softmax(to.relu(r*x))
        sym_string = 'SYMMETRY-BREAKING'
    elif activation == 'sigmoid':
        activation = lambda x: to.sigmoid(x/r)
        sym_string = 'SYMMETRY-PRESERVING'
    elif activation == 'tanh':
        activation = to.tanh
        sym_string = 'SYMMETRY-PRESERVING'
    else:
        raise ValueError('mis-specification')

    for i in tqdm(range(T), desc='PATTERN FORMATION (ORTHOGONAL, %s)'%sym_string):
        Gtilde = to.rfft(input=G, signal_ndim=2, normalized=False, onesided=False)[:,:,:,0] # real FFT
        X = Gtilde*Ctilde
        X = to.cat((X.unsqueeze(-1), to.zeros(X.unsqueeze(-1).size())),dim=-1)
        Gconv = to.ifft(input=X, signal_ndim=2, normalized=False)[:,:,:,0] # inverse real FFT
        H = to.roll(input=Gconv, shifts=(res//2+1,res//2+1), dims=(1,2)) # center zero frequencies

        Hr = H.reshape([Ng, -1])
        Gr = lr*G.reshape([Ng, -1])

        X = Gr@Hr.T # overflows tend to appear here
        oja = (Gr.T@np.tril(X)).T.reshape([Ng,res,res]) # or here
        G += lr*H - oja + lr*activation(G)
        if max_norm:
            G /= G.max() # avoid overflows

        # X = Gr@Hr.T # overflows tend to appear here
        # oja = (Gr.T@np.tril(X)).T.reshape([Ng,res,res]) # or here
        # G += lr * (H - oja + activation(G))
        if np.isnan(G).any():
            print('NaNs detected!')
            return Glast.numpy()
        else:
            Glast = G.clone()
    return G.numpy()




def nonperiodic_ortho_grid_pattern_formation(place_cells, options, max_norm=False, activation='relu'):
    """ orthogonalized multi-cell pattern formation without periodicity, torch implementation
    i.e. instead of convolving an average covariance function across the grid code,
    the grid code is multiplied by the actual covariance matrix """
    raise ValueError('untested')
    if hasattr(options, 'T'):
        T = options.T
    else:
        T = 5000
    if hasattr(options, 'lr'):
        lr = options.lr
    else:
        lr = 8e-3
    if hasattr(options, 'r'):
        r = options.r
    else:
        r = 30
    Ng = options.Ng
    res = options.res
    G = to.as_tensor(np.random.randn(Ng,res,res) * 1e-8).double()
    if not hasattr(place_cells, 'SigmaP'):
        place_cells.compute_covariance()
    SigmaP = to.as_tensor(place_cells.SigmaP).double()

    if activation == 'relu':
        activation = lambda x: to.relu(r*x)
        sym_string = 'SYMMETRY-BREAKING'
    elif activation == 'relu-sm':
        activation = lambda x: F.softmax(to.relu(r*x))
        sym_string = 'SYMMETRY-BREAKING'
    elif activation == 'sigmoid':
        activation = lambda x: to.sigmoid(x/r)
        sym_string = 'SYMMETRY-PRESERVING'
    elif activation == 'tanh':
        activation = to.tanh
        sym_string = 'SYMMETRY-PRESERVING'
    else:
        raise ValueError('mis-specification')

    for i in tqdm(range(T), desc='NON-PERIODIC PATTERN FORMATION (ORTHOGONAL, %s)'%sym_string):
        H = SigmaP@G
        Hr = H.reshape([Ng, -1])
        Gr = G.reshape([Ng, -1])

        X = Gr@Hr.T # overflows tend to appear here
        oja = (Gr.T@np.tril(X)).T.reshape([Ng,res,res]) # or here
        G += lr * (H - oja + activation(G))
        if np.isnan(G).any():
            print('NaNs detected!')
            return Glast.numpy()
        else:
            Glast = G.clone()
    return G.numpy()




def Gaussian(center_ix, sigma, options):
    assert len(center_ix)==2, '2d index'
    assert 0<=center_ix[0]<options.res, 'center out of range'
    assert 0<=center_ix[1]<options.res, 'center out of range'
    np.seterr('ignore')
    pos = np.array(
        np.meshgrid(
            np.linspace(-options.box_width / 2, options.box_width / 2, options.res),
            np.linspace(-options.box_height / 2, options.box_height / 2, options.res),
        )
    ).T.astype(np.float32)

    center_pos = pos[center_ix[0], center_ix[1]]
    norm2 = 10*np.sum(np.abs(center_pos - pos) ** 2, axis=-1) # arbitrary scaling to avoid underflows

    outputs = np.exp(-norm2 / (2 * sigma ** 2))/(np.sqrt(2*np.pi)*sigma)
    outputs /= outputs.sum()
    return outputs


def DoG(center_ix, sigma, surround_scale, options):
    assert len(center_ix)==2, '2d index'
    assert 0<=center_ix[0]<options.res, 'center out of range'
    assert 0<=center_ix[1]<options.res, 'center out of range'
    np.seterr('ignore')
    pos = np.array(
        np.meshgrid(
            np.linspace(-options.box_width / 2, options.box_width / 2, options.res),
            np.linspace(-options.box_height / 2, options.box_height / 2, options.res),
        )
    ).T.astype(np.float32)

    center_pos = pos[center_ix[0], center_ix[1]]
    norm2 = 10*np.sum(np.abs(center_pos - pos) ** 2, axis=-1) # arbitrary scaling to avoid underflows

    outputs = np.exp(-norm2 / (2 * sigma ** 2))/(np.sqrt(2*np.pi)*sigma)
    outputs -= np.exp(-norm2 / (2 * surround_scale * sigma ** 2))/(np.sqrt(2*np.pi*surround_scale)*sigma)
    return outputs


def diff_gen(options, sigma=0.1, surround_scale=1.5, lowrank=False):
    """
    returns a "smooth" diffusion generator based on a difference of gaussians
    lowrank = True implies that Q.shape = (Ng,Np) i.e. Np evenly spaced diffusion meta-states
    """
    if lowrank:
        Q = np.zeros((options.Nx,options.Np))
        step = int(options.res//np.sqrt(options.Np))
        for ix, cx in enumerate(np.arange(0, options.res, step)):
            for iy, cy in enumerate(np.arange(0, options.res, step)):
                center_ix = [cx,cy]
                X = DoG(center_ix, sigma, surround_scale, options)
                X *= -1
                S = X.sum()
                A = np.abs(X).sum()
                X -= np.abs(X)*S/A
                Q[:,ix*step + iy] = X.flatten()
    else:
        Q = np.zeros((options.Nx,options.Nx))
        for ix in tqdm(range(options.Nx), desc='CONSTRUCTING DIFFUSION GENERATOR'):
            center_ix = np.unravel_index(ix, (options.res,options.res))
            X = DoG(center_ix, sigma, surround_scale, options)
            X *= -1
            S = X.sum()
            A = np.abs(X).sum()
            X -= np.abs(X)*S/A
            Q[ix,:] = X.flatten()
    return Q




if run_experiments:
    # Training options and hyperparameters
    class Options:
        pass
    options = Options()

    options.save_dir = 'pattern_formation'
    options.res = 55
    options.Nx = options.res**2
    options.Np = 512              # number of place cells
    options.Ng = 32             # number of grid cells
    options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
    options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
    options.MODEL_type = 'PATFORM'
    options.lr = 1e-4
    options.T = 1000
    options.activation = 'relu'   # recurrent nonlinearity
    options.DoG = True            # use difference of gaussians tuning curves
    options.periodic = False      # trajectories with periodic boundary conditions
    options.norm_cov = False   # normalize translated/averaged spatial covariance (dont use with DoG - autocorrelation signal not strong enough)
    options.gauss_norm = False
    options.box_width = 2.2       # width of training environment
    options.box_height = 2.2      # height of training environment
    # options.run_ID = generate_run_ID_PATFORM(options)

    # NON-ORTHOGONAL SINGLE-CELL PATTERN FORMATION DYNAMICS
    options.Ng = 32
    options.lr = 1e-4
    place_cells = PlaceCells(options)

    # Symmetry-preserving nonlinearity (tanh)
    G = grid_pattern_formation(place_cells, options, activation='tanh')
    plot_ratemaps(G, options.Ng)
    # Symmetry-breaking nonlinearities (relu)
    options.lr = 5e-3
    G = grid_pattern_formation(place_cells, options, activation='relu')
    plot_ratemaps(G, options.Ng)



    # ORTHOGONAL POPULATION PATTERN FORMATION DYNAMICS
    # good run
    options.res = 64
    options.DoG = False
    options.norm_cov = True # True works better with gaussian tuning
    options.gauss_norm = False
    options.Ng = 64
    options.T = 10000 # running for longer is better
    options.lr = 8e-3
    options.r = 25 # increasing the gain, sharpens the grid fields

    # increasing res allows facilitates an increased nG
    # options.res = 128
    # options.DoG = False
    # options.norm_cov = True # True works better with gaussian tuning
    # options.gauss_norm = False
    # options.Ng = 128
    # options.T = 5000 # running for longer is better
    # options.lr = 8e-3
    # options.r = 25 # increasing the gain, sharpens the grid fields

    place_cells = PlaceCells(options)

    # Symmetry-preserving nonlinearity (tanh)
    G = ortho_grid_pattern_formation(place_cells, options, activation='tanh')
    plot_ratemaps(G, options.Ng)

    # Symmetry-breaking nonlinearity (relu)
    G = ortho_grid_pattern_formation(place_cells, options, activation='relu')
    plot_ratemaps(G, options.Ng)

    # Symmetry-preserving nonlinearity (sigmoid)
    G = ortho_grid_pattern_formation(place_cells, options, activation='sigmoid')
    plot_ratemaps(G, options.Ng)

    # combination of symmetry-preserving and symmetry-breaking nonlinearities (relu-sm)
    G = ortho_grid_pattern_formation(place_cells, options, activation='relu-sm')
    plot_ratemaps(G, options.Ng)

    # Symmetry-breaking nonlinearity (relu) with difference-of-gaussians
    options.DoG = True
    options.norm_cov = False # False works better with DoG
    # options.Ng = 32
    # options.lr = 5e-3
    # options.r = 50
    options.Ng = 128 # large Ng leads to a high proportion of plane waves
    options.lr = 5e-3
    options.T = 50000
    options.r = 50/options.Ng # reducing this helps to avoid overflow for large Ng
    place_cells = PlaceCells(options)
    G = ortho_grid_pattern_formation(place_cells, options, activation='relu')
    plot_ratemaps(G, options.Ng)
