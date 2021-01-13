#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import matplotlib.colors as mcol

from collections import namedtuple
from gym.spaces import Box
from gym.spaces import Discrete
from visualization import outline_area

from utils import GWcoords2ix, GWix2coords, GWixvec2coords
from mazelab import BaseEnv, BaseMaze, VonNeumannMotion, Object
from visualization import remove_axes, grid_wrap_nrow, grid_wrap_ncol, get_ncol, nrow, gridspec_kw, figsize, cmap_grid_code, suptitle_fontsize


class RoomsMaze(BaseMaze):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in objects])
        object_type = namedtuple("Objects", map(lambda x: x.name, objects))
        object_type.__new__.__defaults__ = tuple(objects)
        self.objects = object_type()

    @property
    def size(self):
        return self.world_array.shape

    def make_objects(self):
        # differentiating rooms
        start_coords = np.array(GWix2coords(self.world_array,self.start))[np.newaxis,:]
        if self.goal is not None:
            goal_coords = np.array(GWix2coords(self.world_array,self.goal))[np.newaxis,:]
        bneck1_coords = np.array(GWix2coords(self.world_array,self.bnecks[0]))[np.newaxis,:]
        bneck2_coords = np.array(GWix2coords(self.world_array,self.bnecks[1]))[np.newaxis,:]
        bneck3_coords = np.array(GWix2coords(self.world_array,self.bnecks[2]))[np.newaxis,:]
        bneck4_coords = np.array(GWix2coords(self.world_array,self.bnecks[3]))[np.newaxis,:]

        room1 = Object('room1', 0, np.array(mcol.to_rgb(self.info_state.loc[self.room1[-1],'color']))*255, False, np.array(GWixvec2coords(self.world_array,self.room1)))
        room2 = Object('room2', 0, np.array(mcol.to_rgb(self.info_state.loc[self.room2[-1],'color']))*255, False, np.array(GWixvec2coords(self.world_array,self.room2)))
        room3 = Object('room3', 0, np.array(mcol.to_rgb(self.info_state.loc[self.room3[-1],'color']))*255, False, np.array(GWixvec2coords(self.world_array,self.room3)))
        room4 = Object('room4', 0, np.array(mcol.to_rgb(self.info_state.loc[self.room4[-1],'color']))*255, False, np.array(GWixvec2coords(self.world_array,self.room4)))
        bneck1 = Object('bneck1', 0, np.array(mcol.to_rgb(self.info_state.loc[self.bnecks[0],'color']))*255, False, bneck1_coords)
        bneck2 = Object('bneck2', 0, np.array(mcol.to_rgb(self.info_state.loc[self.bnecks[1],'color']))*255, False, bneck2_coords)
        bneck3 = Object('bneck3', 0, np.array(mcol.to_rgb(self.info_state.loc[self.bnecks[2],'color']))*255, False, bneck3_coords)
        bneck4 = Object('bneck4', 0, np.array(mcol.to_rgb(self.info_state.loc[self.bnecks[3],'color']))*255, False, bneck4_coords)
        obstacle = Object('obstacle', 1, np.array(mcol.to_rgb('black'))*255, True, np.stack(np.where(self.world_array == 1), axis=1))
        start = Object('start', 2, np.array(mcol.to_rgb(self.info_state.loc[self.start,'color']))*255, False, start_coords) # red
        if self.goal is not None:
            goal = Object('goal', 3, np.array(mcol.to_rgb(self.info_state.loc[self.goal,'color']))*255, False, goal_coords) # blue
        else:
            goal = Object('goal', 3, np.array(mcol.to_rgb(self.info_state.loc[self.start,'color']))*255, False, [0,0])
        return room1, room2, room3, room4, bneck1, bneck2, bneck3, bneck4, obstacle, start, goal


class Maze(BaseMaze):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        objects = self.make_objects()
        assert all([isinstance(obj, Object) for obj in objects])
        object_type = namedtuple("Objects", map(lambda x: x.name, objects))
        object_type.__new__.__defaults__ = tuple(objects)
        self.objects = object_type()

    @property
    def size(self):
        return self.world_array.shape

    def make_objects(self):
        free = Object('free', 0, np.array(mcol.to_rgb('lightgray'))*255, False, np.stack(np.where(self.world_array == 0), axis=1))
        obstacle = Object('obstacle', 1, np.array(mcol.to_rgb('black'))*255, True, np.stack(np.where(self.world_array == 1), axis=1))
        return free, obstacle


class GridWorldEnv(BaseEnv):
    def __init__(self, maze):
        """
        https://github.com/zuoxingdong/mazelab
        https://github.com/zuoxingdong/mazelab/blob/master/examples/navigation_env.ipynb
        """
        super().__init__()
        self.reward = None
        self.stepcost = None
        self.maze = maze
        self.motions = VonNeumannMotion()
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)] # NORTH, EAST, SOUTH, WEST

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # viz properties
        self.env_lw = 2

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = self.reward
            done = True
        elif not valid:
            reward = -self.reward
            done = False
        else:
            reward = -self.stepcost
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def plot_state_func(self, state_vals, ax=None, annotate=False, interpolation='nearest', cmap=plt.cm.autumn, cbar=False, cbar_label='', node_edge_color=None, arrows=None, mask_color='white', mask_alpha=0., **kwargs):
        """
        FUNCTION: plots state function state_vals on world_array imshow.
        INPUTS: state_vals = state values of function to plot
                ax = figure axis to plot to
                annotate = textualize function values on states
                interpolation = function value interpolation
                cmap = colormap
                cbar = include offset colorbar
                node_edge_color = dummy variable
                arrows = dummy variable
                mask_color/mask_alpha = color/alpha of e.g. inaccessible areas in a maze
                self.env_lw = linewidth of maze outline
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        # variables
        space_shape = self.world_array.shape
        nS = state_vals.size
        FGW = np.ones(space_shape)*np.nan
        FGW_annote = np.ones(space_shape)*np.nan
        IX0,IX1 = np.where(self.world_array==0)
        nSGW = IX0.size
        if nSGW != nS:
            raise ValueError('Wrong number of states')
        S = [] # indices of available states
        for si in range(nS):
            S.append(np.ravel_multi_index((IX0[si],IX1[si]), space_shape))

        for si in range(nS):
            s = S[si]
            FGW[np.unravel_index(s,space_shape)] = state_vals[si]
            FGW_annote[np.unravel_index(s,space_shape)] = state_vals[si]
        FGW = np.ma.masked_values(FGW,np.nan)
        FGW_annote = np.ma.masked_values(FGW_annote,np.nan)
        cmap.set_bad(mask_color, mask_alpha)

        # remove external boundaries
        FGW = FGW[1:,:]
        FGW = FGW[:,1:]
        FGW = FGW[:-1,:]
        FGW = FGW[:,:-1]

        cax = ax.matshow(FGW, cmap=cmap, aspect='auto', interpolation=interpolation, **kwargs)

        # Loop over data dimensions and create text annotations
        if annotate:
            for i in range(FGW.shape[0]):
                for j in range(FGW.shape[1]):
                    val = FGW[i, j]
                    if not np.isnan(val):
                        text = ax.text(j, i, val, ha="center", va="center", color="k")

        # equalize axes and make invisible
        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale(False) # dont scale
        plt.sca(ax)
        if cbar:
            fig = plt.gcf()
            cbar = fig.colorbar(cax, shrink=1.)
            if cbar_label != '':
                cbar.set_label(cbar_label)

        outline_area(mask=self.world_array, ax=ax, color='k', offset=(-1.5,-1.5), lw=self.env_lw)

        return ax


    def get_image(self):
        return self.maze.to_rgb()

    def plot_environment(self, ax=None):
        """
        FUNCTION: Plots grid world environment from self.maze.
        INPUTS: ax for plotting
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        # create image
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_alpha = np.ones((img.shape[0],img.shape[1],4)).astype(np.uint8)*255
        img_alpha[:,:,:3] = img
        ax.matshow(img_alpha)
        ax.set_axis_off()
        return ax

    def identify_states(self):
        """ plots environment and numbers states """
        plt.figure(figsize=(20,20))
        self.plot_state_func(state_vals=np.arange(self.n_state).astype('int'), interpolation='none', annotate=True, ax=plt.gca())
