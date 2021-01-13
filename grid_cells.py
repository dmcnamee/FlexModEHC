#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://github.com/ganguli-lab/grid-pattern-formation
# {Sorscher  B.  Mel  G.  Ganguli  S. \& Ocko  S. A unified theory for the origin of grid cells through the lens of pattern formation. \textit{Advances in Neural Information Processing Systems} (2019).}


import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2

from tqdm import tqdm
from scores import GridScorer


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im

def image_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig

def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    im = image_ratemaps(activations, n_plots, cmap, smooth, width)
    plt.figure(figsize=(16,12))
    plt.imshow(im)
    plt.axis('off')
    return plt.gcf()

def plot_grid_cells(G, options):
      Ng = options.Ng
      res = options.res
      rate_map = G.T # see visualize.compute_ratemaps in grid-pattern-formation package for reasoning
      activations = G.T.reshape(options.Ng,res,res).numpy()

      # start of code from inspect_model
      starts = [0.2] * 10
      ends = np.linspace(0.4, 1.0, num=10)
      box_width=options.box_width
      box_height=options.box_height
      coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
      masks_parameters = zip(starts, ends.tolist())
      scorer = GridScorer(res, coord_range, masks_parameters)

      score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
            *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map, desc='GRID-SCORING RATEMAPS')])

      idxs = np.flip(np.argsort(score_60))
      Ng = options.Ng

      # Plot high grid scores
      n_plot = np.min([128,Ng//3])
      plt.figure(figsize=(16,4*n_plot//8**2))
      rm_fig = image_ratemaps(activations[idxs], n_plot, smooth=True)
      plt.imshow(rm_fig)
      plt.suptitle('Grid scores '+str(np.round(score_60[idxs[0]], 2))
                   +' -- '+ str(np.round(score_60[idxs[n_plot]], 2)),
                  fontsize=16)
      plt.axis('off');

      # Plot medium grid scores
      plt.figure(figsize=(16,4*n_plot//8**2))
      rm_fig = image_ratemaps(activations[idxs[Ng//4:]], n_plot, smooth=True)
      plt.imshow(rm_fig)
      plt.suptitle('Grid scores '+str(np.round(score_60[idxs[Ng//2]], 2))
                   +' -- ' + str(np.round(score_60[idxs[Ng//2+n_plot]], 2)),
                  fontsize=16)
      plt.axis('off');

      # Plot low grid scores
      plt.figure(figsize=(16,4*n_plot//8**2))
      rm_fig = image_ratemaps(activations[np.flip(idxs)], n_plot, smooth=True)
      plt.imshow(rm_fig)
      plt.suptitle('Grid scores '+str(np.round(score_60[idxs[-n_plot]], 2))
                   +' -- ' + str(np.round(score_60[idxs[-1]], 2)),
                  fontsize=16)
      plt.axis('off');



def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res])
    counts  = np.zeros([res, res])

    for index in tqdm(range(n_avg), leave=False, desc='Computing ratemaps'):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        g_batch = model.g(inputs)

        pos_batch = np.reshape(pos_batch, [-1, 2])
        g_batch = np.reshape(tf.gather(g_batch, idxs, axis=-1), (-1, Ng))

        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x <= res and y >=0 and y <= res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos
