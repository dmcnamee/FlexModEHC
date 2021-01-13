#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb
import networkx as nx
import numpy as np
import pandas as pd
import importlib
import itertools

from matplotlib import gridspec
from pprint import pprint



# SETTINGS
overwrite_figures = True
page_width = 8      # A4 width in inches (minus margins)
row_height = 11/4. # A4 height in inches (minus margins) divided into 4 rows
figsize = (page_width, row_height)
nrow = 3
plot_context = 'paper'
plot_style = 'ticks'
palette = 'colorblind'
font_scale = 1.1
label_scale = 1.4


sb.set(context=plot_context, style=plot_style, palette=palette, font='sans-serif', font_scale=font_scale, color_codes=palette)
plt.style.use(['FlexModEHC.mplrc']) # overwrite some custom adaptations
# pprint(sb.plotting_context())
toptitle_fontsize = 45
suptitle_fontsize = 35
color_background_val = 0.5
label_size = label_scale*plt.rcParams['axes.titlesize']
label_weight = plt.rcParams['axes.titleweight']

gridspec_kw = {'left':.01, 'bottom':.1, 'right':.99, 'top':.9, 'wspace':0.6, 'hspace':0.3}
cmap_state_density = plt.cm.bone_r
cmap_spec_density = plt.cm.autumn
cmap_statespaceBG = plt.cm.Greys
cmap_statespaceBG_val = 0.9
cmap_stateseq = plt.cm.cool
cmap_grid_code = plt.cm.jet
cmap_activation_prob = plt.cm.inferno
color_time_covered = 'black'
jitter_state = False
color_diff = 'red'
color_superdiff = 'blue'
color_turb = 'purple'
color_acmin = 'darkorange'

suptitle_yshift = 1.03

# graph variables
text_font_size = '70pt'
min_node_size = 150
max_node_size = 220
min_edge_size = 5
max_edge_size = 80
node_sizes = None
edge_sizes = None
min_node_lw = 1
max_node_lw = 10
node_size = (min_node_size+max_node_size)/2.
edge_size = (min_edge_size+max_edge_size)/2.
cmap_edge = None
color_index_edge = None


# helper functions
def label_lines(ax):
    plt.sca(ax)
    labelLines(ax.get_lines(),zorder=2.5)

def label_panel(ax, label='A', x=-0.15, y=1.1, fontweight=label_weight, fontsize=label_size):
    ax.text(x, y, label, transform=ax.transAxes, fontweight=fontweight, fontsize=fontsize, va='top', ha='right')

def label_panels(axes, x=-0.15, y=1.1, fontweight=label_weight, fontsize=label_size):
    from collections import Iterable
    from string import ascii_uppercase
    if isinstance(axes[0],Iterable):
        axes = list(itertools.chain.from_iterable(axes))
    for ix, ax in enumerate(axes):
        label_panel(ax, label=ascii_uppercase[ix], x=x, y=y, fontweight=fontweight, fontsize=fontsize)

def despine_panels(axes):
    from collections import Iterable
    from string import ascii_uppercase
    if isinstance(axes[0],Iterable):
        axes = list(itertools.chain.from_iterable(axes))
    for ix, ax in enumerate(axes):
        sb.despine(ax=ax)

def postproc_panels(ax, label=True):
    if label:
        label_panels(ax)
    despine_panels(ax)

def add_jitter(X, std=0.1):
    """
    FUNCTION: convenience function to jitter state coordinates.
    """
    n_steps = X.shape[0]
    n_dim = X.shape[1]
    for d in range(n_dim):
        X[:,d] = X[:,d] + np.random.normal(0,std,n_steps)
    return X


def save_figure(fig, figdir, fname_base, file_ext='.pdf'):
    """Convenience function to save figures."""
    if figdir is not None:
        if not overwrite_figures:
            rng = '_'+random_filename(base='')
        else:
            rng = ''
        fname = fname_base+rng+file_ext
        fname_lowres = fname_base+rng+'_LOWRES'+file_ext
        fig.savefig(os.path.join(figdir,fname), transparent=False, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(figdir,fname_lowres), transparent=False, dpi=100, bbox_inches='tight')


def remove_axes(ax):
    sb.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def grid_wrap_ncol(total, wrap_row=None):
    if wrap_row is not None:
        n_row = min(total, wrap_row)
        n_col = 1 + (total - 1)//wrap_row
    else:
        n_row = total
        n_col = 1
    return n_row,n_col

def grid_wrap_nrow(total, wrap_col=None):
    if wrap_col is not None:
        n_col = min(total, wrap_col)
        n_row = 1 + (total - 1)//wrap_col
    else:
        n_col = total
        n_row = 1
    return n_row,n_col

def get_ncol(n):
    return np.ceil(n/float(nrow)).astype('int')

def random_filename(base=''):
    import uuid
    return base + '' + uuid.uuid4().hex[:8]

def plot_state_func(F, world_array=None, ax=None, no_vals=False, hist=False,
                        annotate=False, cmap=plt.cm.autumn, cbar=False, cbar_label='', **kwargs):
    """
    FUNCTION: plots state function F on world_array imshow.
    INPUTS: F = function to plot
            ax = figure axis to plot to
            world_array = grid world_array to serve as background
            no_vals = don't plot function F on gridworld
            hist = also plot function value histogram
            annotate = textualize function values on states
            cmap = colormap
            cbar = include offset colorbar
            cbar_label = label for colorbar
    NOTES: no_vals=True implies that plot_state_func just plots a gridworld environment
    """
    # variables
    space_shape = world_array.shape
    nS = F.size
    FGW = np.ones(space_shape)*np.nan
    FGW_annote = np.ones(space_shape)*np.nan
    IX0,IX1 = np.where(world_array==0)
    nSGW = IX0.size
    if nSGW != nS:
        raise ValueError('Wrong number of states')
    S = [] # indices of available states
    for si in range(nS):
        S.append(np.ravel_multi_index((IX0[si],IX1[si]), space_shape))

    if ax is None:
        if hist is True:
            fig,axs = plt.subplots(1,2)
            ax = axs[0]
            axs[1].hist(F,normed=True)
            axs[1].set_title('Function Histogram')
        else:
            fig = plt.figure()
            ax = plt.gca()

    for si in range(nS):
        s = S[si]
        if not no_vals:
            FGW[np.unravel_index(s,space_shape)] = F[si]
            FGW_annote[np.unravel_index(s,space_shape)] = F[si]
        else:
            FGW[np.unravel_index(s,space_shape)] = color_background_val
            FGW_annote[np.unravel_index(s,space_shape)] = F[si]
    FGW = np.ma.masked_values(FGW,np.nan)
    FGW_annote = np.ma.masked_values(FGW_annote,np.nan)
    cmap.set_bad('black',1.)

    # cax = sb.heatmap(FGW, cmap=cmap, aspect='auto', ax=ax)
    cax = ax.imshow(FGW, cmap=cmap, aspect='auto', **kwargs)

    # Loop over data dimensions and create text annotations
    if annotate:
        for i in range(FGW.shape[0]):
            for j in range(FGW.shape[1]):
                val = FGW_annote[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, val, ha="center", va="center", color="k")

    # ax.set_title('Function Value')
    remove_axes(ax)
    ax.axis('equal')
    if cbar:
        fig = plt.gcf()
        cbar = fig.colorbar(cax, shrink=0.8)
        if cbar_label != '':
            cbar.set_label(cbar_label)
    plt.tight_layout()
    return ax



def plot_propagation_comparison(PROPS, figdir=None):
    """
    FUNCTION: Plots diagnostic comparisons across multiple PROPs.
    NOTES: deprecated, plotting functions distributed to EXPLORER class
    """
    nprop = len(PROPS)
    # labels = [PROP.label for PROP in PROPS]
    fig, axes = plt.subplots(2, 2, figsize=(12,12), sharey=False)
    for propn in range(nprop):
        plot_grid_gains(PROPS[propn],ax=axes[0][0])
        plot_msd(PROPS[propn],loglog=False,ax=axes[0][1])
        axes[0][1].legend().set_visible(False)
        plot_coverage(PROPS[propn],ax=axes[1][0])
        axes[1][0].legend().set_visible(False)
        plot_exploration_efficiency(PROPS[propn],ax=axes[1][1])
        axes[1][1].legend().set_visible(False)

    axes[0][1].set(xscale='log',yscale='log')
    axes[0][1].set_xlim([np.log(1),axes[0][1].get_xlim()[1]])
    axes[0][1].set_ylim([np.log(1),axes[0][1].get_ylim()[1]])

    plt.tight_layout()
    save_figure(fig=fig,figdir=figdir,fname_base='prop_comparison')


def plot_grid_gains_comparison(PROPS, figdir=None):
    """
    FUNCTION: Plots diagnostic comparisons across multiple PROPs.
    """
    nprop = len(PROPS)
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    for propn in range(nprop):
        plot_grid_gains(PROPS[propn],ax=ax)
    plt.tight_layout()
    save_figure(fig=fig,figdir=figdir,fname_base='grid_gains')


def plot_diagonal_line(ax, start_pos='BL', color=None):
    if start_pos == 'BL':
        ax.plot(ax.get_xlim(),ax.get_ylim(), ls='--', color=color)
    elif start_pos == 'UL':
        ax.plot(ax.get_xlim().reverse(),ax.get_ylim(), ls='--', color=color)
    elif start_pos == 'BR':
        ax.plot(ax.get_xlim(),ax.get_ylim().reverse(), ls='--', color=color)
    elif start_pos == 'UR':
        ax.plot(ax.get_xlim().reverse(),ax.get_ylim().reverse(), ls='--', color=color)


def plot_trajectory_generation(PROP, horizontal=False, jitter_state=jitter_state, figdir=None):
    """
    FUNCTION: plots generated trajectory along with scaled grid cells and state posteriors.
    NOTE: High-level plot.
    """
    wspace = None
    hspace = None
    size1 = 40
    size2 = 12
    panel_yshift = 0.92
    label = PROP.label
    nplot = np.min([9,PROP.nstate])
    ix_per_row = get_ncol(nplot)
    if horizontal:
        fig = plt.figure(figsize=(size1, size2))
        outer = gridspec.GridSpec(1,3)
    else:
        fig = plt.figure(figsize=(size2, size1))
        outer = gridspec.GridSpec(3,1)

    # scaled grid kernels
    ax_outer = plt.subplot(outer[0]); ax_outer.axis('off')
    inner = gridspec.GridSpecFromSubplotSpec(ix_per_row, ix_per_row, wspace=wspace, hspace=hspace, subplot_spec=outer[0])
    FK = PROP.freq_basis.real
    vmin = FK[:,:nplot].flatten().min()
    vmax = FK[:,:nplot].flatten().max()
    for i in range(ix_per_row):
        for j in range(ix_per_row):
            ix = ix_per_row*i+j
            ax = plt.Subplot(fig, inner[i,j])
            fig.add_subplot(ax)
            plot_freq_kernel(PROP,ix=ix,ax=ax,vmin=vmin,vmax=vmax)
            ax.axis('equal')
            # ax.set_title('Grid cell tuning %i'%ix, y=panel_yshift)
    ax_outer.set_title('Grid representations',y=suptitle_yshift, fontsize=suptitle_fontsize)

    # plot state posteriors
    ax_outer = plt.subplot(outer[1]); ax_outer.axis('off')
    inner = gridspec.GridSpecFromSubplotSpec(ix_per_row, ix_per_row, wspace=wspace, hspace=hspace, subplot_spec=outer[1])
    for i in range(ix_per_row):
        for j in range(ix_per_row):
            ix = ix_per_row*i+j
            ax = plt.Subplot(fig, inner[i,j])
            fig.add_subplot(ax)
            plot_state_density(PROP,ix=ix,ax=ax,cbar=False)
            ax.axis('equal')
            ax.set_title('State density on step %i'%ix, y=panel_yshift)
    ax_outer.set_title('Output state-space densities',y=suptitle_yshift,fontsize=suptitle_fontsize)

    # plot trajectory
    ax = plt.Subplot(fig, outer[2])
    fig.add_subplot(ax)
    plot_trajectory(PROP, jitter_state=jitter_state, plot_state_seq=True, ax=ax, cbar=False, figdir=None)
    ax.set_title('Trajectory sample', fontsize=suptitle_fontsize)

    # plt.suptitle(label, fontsize=toptitle_fontsize)
    fig.tight_layout()
    save_figure(fig=fig,figdir=figdir,fname_base='trajgen')


def plot_wfill(y, e, x=None, ax=None, color='k', alpha=0.2, **kwargs):
    if ax is not None:
        plt.sca(ax)
    if x is None:
        x = np.arange(y.size)
    else:
        y = y[:x.size]
        e = e[:x.size]
    plt.plot(x, y, color=color, **kwargs)
    plt.fill_between(x, y, y+e, color=color, alpha=alpha)
    plt.fill_between(x, y, y-e, color=color, alpha=alpha)


def outline_area(mask, ax, area_value=0, color='k', offset=(-1.5,-1.5), lw=3):
    """
    FUNCTION: plots an outline of contiguous components of a mask
    INPUTS: mask = to plot
            ax = axis to plot to
            area_value = value of masked area
            color = of outline
            offset = where is the origin
            lw = 3, linewidth
    NOTES: from https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    """
    # create a boolean image map which has trues only where maskimg[x,y] == area_value
    mapimg = (mask == area_value)

    # a vertical line segment is needed, when the pixels next to each other horizontally
    # belong to different groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    # segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    # segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    ax.plot(segments[:,0]+offset[0], segments[:,1]+offset[1], color=color, linewidth=lw, clip_on=False, zorder=100)


# set the colormap and centre the colorbar
# http://chris35wills.github.io/matplotlib_diverging_colorbar/
class MidpointNormalize(mpl.colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def rgb(im, cmap='jet', smooth=True):
    """from https://github.com/ganguli-lab/grid-pattern-formation"""
    import cv2
    cmap = plt.cm.get_cmap(cmap)
    # np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    # im = np.uint8(im * 255)
    return im


def change_seaborn_bar_width(ax, new_value):
    # https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def interpolate_curve(x,y, boundary_correction=True):
    from scipy.interpolate import make_interp_spline
    if boundary_correction:
        xnew = np.linspace(x.min(), x.max()+1, 300)
        spl = make_interp_spline(list(x)+[x.max()+1], list(y)+[0], k=3)
        return xnew, spl(xnew)
    else:
        xnew = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(list(x)+[x.max()], list(y), k=3)
        return xnew, spl(xnew)
