#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import gym
import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.generators import triangular_lattice_graph
from collections import namedtuple
from gym.spaces import Box
from gym.spaces import Discrete
from gym import spaces
from gym.utils import seeding
from copy import deepcopy

import holoviews as hv
from utils import create_tensor, row_norm
from mazes import RoomsMaze, Maze, GridWorldEnv
from visualization import (
    text_font_size,
    node_size,
    edge_size,
    color_index_edge,
    cmap_edge,
    remove_axes,
)
from utils import (
    GWcoords2ix,
    GWix2coords,
    GWixvec2coords,
    Amat,
    l1_normalize_rows,
    pol2cart,
    cart2pol,
    pos_dict,
    rotate_around_point,
)
from networkx.drawing.nx_agraph import graphviz_layout

hv.extension("bokeh", "matplotlib")


class GraphEnv(gym.Env):
    """
    Simple openai-gym environment wrapper.
    Instructions: initialize then set the reward function with goal_func.
    May want to customize __init__, _access_matrix, _node_info,...
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, viz_scale=1.0):
        self.__name__ = "generic-env"
        self.__type__ = "graph"
        self.goal_absorb = False
        self.stay_actions = False
        self.viz_scale = viz_scale
        self._state_information()
        self._transition_information()
        self.set_viz_scheme()
        self.fname_graph = "figures/graph.png"
        self.n_dim = 2  # assume 2D embedding

        # action space defined as number of states
        self.action_space = spaces.Discrete(self.n_state)
        self.naction_max = (self.A != 0).sum(axis=1).max()

        # discrete set of states
        self.observation_space = spaces.Discrete(self.n_state)

        # initialize
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.start
        self.lastaction = None
        return self.state

    def step(self, action):
        if self.A[self.state, action]:
            # available transition
            reward = self.R[self.state, action]
            self.state = action
            if self.state == self.goal:
                done = True
            else:
                done = False
            return self.state, reward, done, {}
        else:
            done = False
            reward = -self.stepcost
            return self.state, reward, done, {}

    def congruent(self, env):
        """
        FUNCTION: Checks if env is a congruent environment to self.
        True if underlying graphs are isomorphic.
        NOTES: More checks in future.
        """
        # return nx.is_isomorphic(self.G, env.G)
        return np.all(self.A == env.A)

    def analogous_policy(self, T, fill_prob=0.001):
        """
        FUNCTION: Returns the policy closest to T that is compatible with the current state-space structure (if possible).
        INPUTS: T = policy to be adapted
                fill_prob = prob of "new" available transitions
        NOTES: Policy may need to be adapted based on goal-state convention (STAY/RESET).
        """
        T[self.A == 0] = 0  # remove impossible transitions
        T[(T == 0) & (self.A == 1)] = fill_prob  # add possible transitions
        return row_norm(T)

    def set_info_state(self, state, type=None, color=None, label=None, **kwargs):
        """Set info state on a per-state basis."""
        if type is not None:
            self.info_state.loc[state, "type"] = type
        if color is not None:
            self.info_state.loc[state, "color"] = color
            self.set_palette()
        if label is not None:
            self.info_state.loc[state, "label"] = label
        for key, value in kwargs.items():
            self.info_state.loc[state, key] = value

    def _check_statespace(self):
        """Performs various checks to ensure that state-space definition is valid."""
        if not nx.is_strongly_connected(nx.DiGraph(self.G)):
            print("ENVIRONMENT: state-space is not connected.")

    def goal_func(
        self,
        reward=10,
        stepcost=1,
        goals=3,
        goal_absorb=False,
        stay_actions=True,
        goal_remain=False,
    ):
        """
            FUNCTION: Defines reward function.
            INPUTS: reward = reward for transition to a goal state
                    stepcost = cost for transitioning to a non-goal state
                    goals = goal states (can choose several)
                    goal_absorb = absorb at goals (not yet implemented in PPP)
                    stay_actions = False implies dissipative MDP without option to stay at states
                                   True implies dissipative MDP with option to stay at states
                    goal_remain = agent forced to remain at goal
                    goal_absorb/stay_actions/goal_remain = False, default is to restart episode (wormhole from goal to start)
        """
        if not hasattr(goals, "__len__"):
            goals = [goals]
        assert np.all(np.array(goals) < self.n_state)
        self.stay_actions = stay_actions
        self.goal_absorb = goal_absorb
        self.goal_remain = goal_remain
        self.reward = reward
        self.stepcost = stepcost
        self.goals = goals

        # modify accessibility matrix self.A
        self.A_adj = self.A.copy()
        for g in self.goals:
            if self.goal_absorb:
                self.A[g, :] = 0  # no edges available from goal state (thus absorbing)
            elif self.stay_actions:
                np.fill_diagonal(
                    self.A, 1
                )  # option to stay at any state (as opposed to reset)
            elif self.goal_remain:
                self.A[g, :] = 0
                self.A[g, g] = 1
            else:
                self.A[g, :] = 0
                self.A[g, self.start] = 1  #  dissipative MDP (forced reset to start)
            self._set_state_forced()
            self.info_state.loc[g, "opt_path"] = "Goal"
            self.info_state.loc[g, "label"] = "G"
            self.info_state.loc[g, "type"] = "Goal"
        # reward function
        self.R = create_tensor((self.n_state, self.n_state), fill_value=-np.inf)
        for i in range(self.n_state):
            for j in range(self.n_state):
                if self.A[i, j] != 0:
                    if j in self.goals:
                        self.R[i, j] = reward
                    else:
                        self.R[i, j] = -stepcost
        if not self.goal_absorb and not self.stay_actions and not self.goal_remain:
            # no restart cost if goal resetting
            for goal in self.goals:
                self.R[goal, self.start] = 0
        self._compute_reward_per_state()
        self.info_state.loc[self.start, "opt_path"] = "Start"
        self.info_state.loc[self.start, "label"] = "S"
        self.info_state.loc[self.start, "type"] = "Start"
        self._label_shortest_paths()
        self._set_graph()

    def distance(self, state1, state2, interval_size=1.0):
        """distance between state1 and state2 = shortest_path_length(A) x interval_size"""
        if not hasattr(self, "shortest_n_steps"):
            self.shortest_n_steps = dict(nx.shortest_path_length(self.G))
        return self.shortest_n_steps[state1][state2] * interval_size

    def _compute_reward_per_state(self):
        """
        FUNCTION: Environment reward functions R is defined per transition.
                  This function computes a state-dependent reward function by marginalizing
                  over a random policy.
        INPUTS: self.R
        OUTPUTS: self.R_state
        NOTES: Transition reward is assigned to outcome state.
        """
        self.R_state = self.R.mean(axis=1)

    def _set_graph(self, X=None):
        """
        FUNCTION: Converts X to a networkx digraph self.G
        NOTES: A_adj is default
        """
        if X is None:
            X = self.A_adj
        self.G = nx.DiGraph(X)
        self._check_statespace()

    def stoch_mat_to_trans_weights(self):
        """converts stochastic matrix to graph edge weights and info_transition"""
        if hasattr(self, "T"):
            if hasattr(self, "G"):
                for edge in self.G.edges:
                    s1 = edge[0]
                    s2 = edge[1]
                    self.G.edges[s1, s2]["prob"] = self.T[s1, s2]
            for ix in range(self.info_transition.shape[0]):
                s = self.info_transition.loc[ix, "source"]
                t = self.info_transition.loc[ix, "target"]
                self.info_transition.loc[ix, "prob"] = self.T[s, t]

    def _set_graph_from_trans_attr(self, attr="prob"):
        edgesdf = self.info_transition
        self.G = nx.from_pandas_edgelist(
            df=edgesdf,
            source="source",
            target="target",
            edge_attr=attr,
            create_using=nx.DiGraph(),
        )
        remove_list = [
            edge for edge in self.G.edges() if self.G.edges[edge[0], edge[1]][attr] == 0
        ]
        self.G.remove_edges_from(remove_list)

    def _pos_dict(self, xymat=None):
        """Convert from xy matrix to pos_dict object used by networkx."""
        if xymat is None:
            xymat = self.xy
        pos = pos_dict(xymat)
        self.pos = pos
        return pos

    def _label_shortest_paths(self):
        """
        FUNCTION: Identifies shortest path from start to goal.
        NOTES: For multiple goals, solves for the nearest goal.
               For multiple shortest paths of equal length, records them all.
        """
        self._set_graph()
        if len(self.goals) > 1:
            # find goal with shortest path length
            current_shortest_length = np.inf
            for g in self.goals:
                goal_shortest_length = nx.shortest_path_length(
                    self.G, source=self.start, target=g
                )
                if goal_shortest_length < current_shortest_length:
                    current_shortest_length = goal_shortest_length
                    goal_shortest = g
        else:
            goal_shortest = self.goals[0]
        paths = list(
            nx.all_shortest_paths(self.G, source=self.start, target=goal_shortest)
        )
        for path in paths:
            for ix, state in enumerate(path):
                self.info_state.loc[state, "opt_path_bool"] = True
                if self.info_state.loc[state, "opt_path"] not in ["Start", "Goal"]:
                    self.info_state.loc[state, "opt_path"] = "Via state"
                self.info_state.loc[state, "opt_path_pos"] = ix

    def _info_goal(self):
        """
        Record nearest goal to a state as well as the distance to that goal.
        Adapts state color scheme to reflect task-orientation in "colors_task" scheme.
        """
        self._set_graph()
        goals = self.goals
        for state in range(self.n_state):
            shortest_lengths = nx.shortest_path_length(self.G, source=state)
            goal_shortest_lengths = np.array([shortest_lengths[i] for i in goals])
            self.info_state.loc[state, "goal_nearest"] = goals[
                goal_shortest_lengths.argmin()
            ]
            self.info_state.loc[state, "goal_dist"] = goal_shortest_lengths.min()

    def _set_state_forced(self):
        """Record states at which policy is forced"""
        self.info_state["forced"] = self.A.sum(1) <= 1

    def _state_information(self):
        """sets information about state including modules/colors etc"""
        self.info_state = pd.DataFrame(
            index=pd.Index(range(self.n_state), name="state")
        )
        self._set_state_forced()
        self.info_state["type"] = "Other"
        self.info_state["goal_nearest"] = -1
        self.info_state["goal_dist"] = np.nan
        self.info_state["opt_path"] = "Not on optimal path"
        self.info_state["opt_path_pos"] = np.nan
        self.info_state["opt_path_bool"] = False
        self.info_state["label"] = (self.info_state.index + 1).astype(
            "str"
        )  #  label states starting at 1
        self.info_state["x"] = np.nan
        self.info_state["y"] = np.nan
        self.info_state = self.info_state.astype(
            dtype={
                "forced": "bool",
                "type": "str",
                "goal_nearest": "int",
                "goal_dist": "float",
                "opt_path": "str",
                "opt_path_pos": "float",
                "opt_path_bool": "bool",
                "x": "float",
                "y": "float",
            }
        )

    def _transition_information(self):
        """sets information about transitions including e.g. weights"""
        if hasattr(self, "G"):
            self.info_transition = nx.to_pandas_edgelist(self.G)
        elif hasattr(self, "W"):
            self.G = nx.DiGraph(self.W)
            self.info_transition = nx.to_pandas_edgelist(self.G)
        elif hasattr(self, "A"):
            self.G = nx.DiGraph(self.A)
            self.info_transition = nx.to_pandas_edgelist(self.G)
        else:
            raise ValueError("Need source for transition information")

    def _define_state_coordinates(self):
        """copies self.xy generated by self._node_info to self.info_state['x','y']"""
        assert hasattr(self, "xy"), "xy node positions unavailable"
        self.info_state["x"] = self.xy[:, 0]
        self.info_state["y"] = self.xy[:, 1]

    def _retrieve_state_coordinates(self, state):
        """returns graph ambient space coordinates, state can be a state int of list/array of states"""
        if hasattr(state, "__len__"):
            return (
                self.info_state.loc[state, ["x", "y"]]
                .values.reshape((len(state), 2))
                .squeeze()
            )
        else:
            return self.info_state.loc[state, ["x", "y"]].values.flatten()

    def set_palette(self, var=None):
        """Copies info_state.color information to palette grouped by var."""
        if var is None:
            var = "hue"
        self.viz_kwargs["palette"] = (
            self.info_state.groupby(self.viz_kwargs[var])
            .apply(lambda x: x.loc[x.index[0], "color"])
            .to_dict()
        )
        self.viz_kwargs_lines["palette"] = self.viz_kwargs["palette"]
        self.viz_kwargs_markers["palette"] = self.viz_kwargs["palette"]

    def set_viz_scheme(
        self, alphas=[0.3, 1.0], sizes_lines=(3, 1), sizes_markers=(300, 200)
    ):
        """
        FUNCTION: Sets plotting variables according to viz_scheme and info_state
        INPUTS: alphas = transparency values for minor and major plot components respectively
                sizes_lines = range of line sizes (inverted for "goal_dist" by default)
                sizes_markers = range of marker sizes (inverted for "goal_dist" by default)
        """
        self.viz_scheme = "default"
        size_lines = "goal_dist"
        size_markers = "KL_prior"
        dashes = {
            "Bottleneck": (2, 2, 10, 2),
            "Start": "",
            "Goal": "",
            "Switch": "",
            "Via": "",
            "Other": "",
        }
        markers = {
            "Bottleneck": "P",
            "Start": "o",
            "Goal": "o",
            "Switch": "o",
            "Via": "o",
            "Other": ".",
        }
        style_order = ["Bottleneck", "Start", "Goal", "Switch", "Via", "Other"]
        hue = "state"
        hue_order = None

        self.viz_kwargs = {
            "legend": None,
            "units": "state",
            "hue": hue,
            "hue_order": hue_order,
            "style": "type",
            "style_order": style_order,
            "estimator": None,
        }
        self.viz_kwargs_markers = {
            **self.viz_kwargs,
            "size": size_markers,
            "sizes": sizes_markers,
            "markers": markers,
        }
        self.viz_kwargs_lines = {
            **self.viz_kwargs,
            "size": size_lines,
            "sizes": sizes_lines,
            "dashes": dashes,
        }
        self.state_type_major = [
            "Bottleneck",
            "Start",
            "Goal",
            "Via",
            "Switch",
        ]
        self.state_type_alphas = {
            "minor": alphas[0],
            "major": alphas[1],
        }

        # construct palettes
        if self.viz_scheme == "default":
            self.color_palette = ["grey"]
            self.info_state.loc[:, "color_index"] = 0
            self.info_state["color"] = self.info_state.color_index.apply(
                lambda x: self.color_palette[int(x)]
            )
        self.info_state = self.info_state.astype(
            dtype={"color_index": "int", "color": "str"}
        )
        self.set_palette()
        if hasattr(self, "goals"):
            self._info_goal()

    def plot_graph(self, width=2000, height=2000, dpi=300):
        """
        FUNCTION: Saves a state-space graph plot.
        """
        # color settings
        node_color = "color"

        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition

        # other settings
        if self.__type__ in ["decision-tree"]:
            directed = True
        else:
            directed = False

        nodes = hv.Nodes(
            data=nodesdf, kdims=["x", "y", "state"], vdims=["type", "color", "label"]
        )
        graph = hv.Graph(
            data=(edgesdf, nodes),
            kdims=["source", "target"],
            vdims=["weight"],
            label=self.__name__,
        )
        labels = hv.Labels(nodes, ["x", "y"], "label").opts(
            text_font_size=text_font_size,
            text_color="black",
            show_frame=False,
            toolbar="disable",
        )

        graph.opts(
            title="",
            directed=directed,
            padding=0.1,
            bgcolor="white",
            width=width,
            height=height,
            show_frame=False,
            xaxis=None,
            yaxis=None,
            node_size=node_size,
            node_color=node_color,
            edge_line_width=edge_size,
            edge_color_index=color_index_edge,
            edge_cmap=cmap_edge,
            edge_line_color="black",
            toolbar="disable",
        )
        graph = graph * labels
        hv.save(graph, filename=self.fname_graph, backend="bokeh", dpi=dpi)

    def plot_stochastic_matrix(self):
        """Plots stochastic matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.T, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("stochastic matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_policy_matrix(self):
        """Plots stochastic matrix"""
        assert hasattr(self, "PI"), "no policy set"
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.PI, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("policy matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_access_matrix(self):
        """Plots accessibility matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.A, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("access matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_adjacency_matrix(self):
        """Plots accessibility matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.A_adj, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_environment(self, X=None, ax=None, figsize=(12, 12)):
        """
        FUNCTION: plot environment graph
        """
        if X is not None:
            self._set_graph(X=X)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.axes(ax)
        self.draw_graph(ax=ax)

    def draw_graph(self, with_labels=False, ax=None):
        """uses networkx to draw graph"""
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        pos = self._pos_dict()
        edge_weights = nx.get_edge_attributes(self.G, "prob").values()
        edge_weights = [10 * e for e in edge_weights]
        nx.draw(
            self.G,
            pos,
            node_size=30,
            alpha=0.7,
            node_color="black",
            width=0,
            with_labels=with_labels,
            font_size=24,
            ax=ax,
            arrows=False,
        )
        nx.draw_networkx_edges(
            G=self.G, pos=pos, width=edge_weights, alpha=0.6, arrows=False, ax=ax
        )
        plt.axis("equal")
        plt.axis("off")

    def plot_state_func(
        self,
        state_vals,
        vlims=[None, None],
        ax=None,
        annotate=False,
        cmap=plt.cm.autumn,
        cbar=False,
        cbar_label="",
        node_edge_color="black",
        **kwargs
    ):
        """
        FUNCTION: plots state function state_vals on world_array imshow.
        INPUTS: state_vals = state values of function to plot
                vlims = [vmin,vmax] value range
                ax = figure axis to plot to
                annotate = textualize function values on states
                cmap = colormap
                cbar = include offset colorbar
                cbar_label = label for colorbar
        """
        if ax is None:
            ax = self.plot_environment(ax=ax)
        state_vals_dict = {}
        for ix, state in enumerate(self.info_state.index):
            state_vals_dict[state] = state_vals[ix]
        nx.set_node_attributes(G=self.G, name="state_val", values=state_vals_dict)
        node_colors = [self.G.nodes[n]["state_val"] for n in self.G.nodes]
        ec = nx.draw_networkx_edges(G=self.G, pos=self.pos, ax=ax, **kwargs)
        nc = nx.draw_networkx_nodes(
            G=self.G,
            pos=self.pos,
            node_size=300,
            node_color=node_colors,
            cmap=cmap,
            ax=ax,
            **kwargs
        )
        if node_edge_color is not None:
            nc.set_edgecolor(node_edge_color)
            nc.set_linewidth(1)
        # create text annotations
        if annotate:
            for state in range(self.n_state):
                x = self.xy[state, 0]
                y = self.xy[state, 1]
                state_val = state_vals[state]
                if not np.isnan(state_val):
                    text = ax.text(x, y, state_val, ha="center", va="center", color="k")
        remove_axes(ax)
        ax.axis("equal")
        if cbar:
            fig = plt.gcf()
            cbar = fig.colorbar(nc, shrink=0.6, orientation="horizontal", pad=0)
            if cbar_label != "":
                cbar.set_label(cbar_label)
        return ax


class RoomWorld(GridWorldEnv, GraphEnv):
    """
    Room world based on Mazelab.
    INPUTS: maze = numpy array indicating available/unavailable states
            start/goal/name/reward/stepcost/goal_absorb/stay_actions as in other envs
    NOTES: takes advantage of MazeLab visualization
    """

    def __init__(
        self,
        start=None,
        goal=None,
        wormhole=None,
        n_rooms=4,
        scale=3,
        name="",
        reward=10,
        stepcost=1,
        goal_absorb=False,
        stay_actions=True,
        goal_remain=False,
    ):
        assert n_rooms == 4, "Can only do four rooms for now."
        if wormhole is not None:
            assert (
                len(wormhole) == 2
            ), "Single wormhole should have a single entrance and single exit."
        self.n_rooms = n_rooms
        self.scale = scale
        self.start_center_TL = int(
            np.ceil(self.scale / 2.0 - 1) * self.scale + np.ceil(self.scale / 2.0 - 1)
        )  #  center of top-left room
        if start == None:
            start = self.start_center_TL
        self.start = start
        self.goal = goal
        self.wormhole = wormhole
        if n_rooms == 4:
            self.world_array = self.gridworld_4room(scale=scale)
        else:
            raise ValueError("Not coded up")
        self._access_matrix()
        self._classify_states()
        # viz properties
        self.env_lw = 2

        GraphEnv.__init__(self)
        if goal is not None:
            self.goal_func(
                reward=reward,
                stepcost=stepcost,
                goals=goal,
                goal_absorb=goal_absorb,
                stay_actions=stay_actions,
                goal_remain=goal_remain,
            )
        self.maze = RoomsMaze(
            world_array=self.world_array,
            room1=self.states_room1,
            room2=self.states_room2,
            room3=self.states_room3,
            room4=self.states_room4,
            bnecks=self.states_bnecks,
            wormhole=self.wormhole,
            info_state=self.info_state,
            start=self.start,
            goal=self.goal,
        )
        self._state_information()
        self._node_info()
        GridWorldEnv.__init__(self, maze=self.maze)
        self.__type__ = "roomworld"
        self.__name__ = self.__type__ + "_" + name

    def gridworld_4room(self, scale=3):
        nG = scale * 2 + 3
        holeV = np.ceil(scale / 2.0).astype("int")
        holeH = np.ceil(scale / 2.0).astype("int")
        world_array = np.zeros((nG, nG)).astype("float")
        world_array[0, :] = 1.0  #  top wall
        world_array[:, 0] = 1.0  # left wall
        world_array[nG - 1, :] = 1.0  # bottom wall
        world_array[:, nG - 1] = 1.0  # right wall

        world_array[scale + 1, :] = 1.0  # horizontal wall
        world_array[scale + 1, holeH] = 0.0  # horizontal hole 1
        world_array[scale + 1, scale + 1 + holeH] = 0.0  # horizontal hole 2
        world_array[:, scale + 1] = 1.0  # vertical wall
        world_array[holeV, scale + 1] = 0.0  # vertical hole 1
        world_array[scale + 1 + holeV, scale + 1] = 0.0  # vertical hole 2
        return world_array

    def _access_matrix(self):
        self.A = Amat(self.world_array)
        self.A_adj = Amat(self.world_array)  # pristine copy of adjacency matrix
        if self.wormhole is not None:
            # add wormhole
            self.A[self.wormhole[0], self.wormhole[1]] = 1
        self.n_state = self.A.shape[0]

    def _classify_states(self):
        self.states_room1 = []
        self.states_room2 = []
        self.states_room3 = []
        self.states_room4 = []
        self.states_bnecks = []
        # room 1
        for x in range(1, 1 + self.scale):
            for y in range(1, 1 + self.scale):
                self.states_room1.append(GWcoords2ix(self.world_array, [x, y]))
        # room 2
        for x in range(1, 1 + self.scale):
            for y in range(1 + self.scale + 1, 1 + self.scale + 1 + self.scale):
                self.states_room2.append(GWcoords2ix(self.world_array, [x, y]))
        # room 3
        for x in range(1 + self.scale + 1, 1 + self.scale + 1 + self.scale):
            for y in range(1, 1 + self.scale):
                self.states_room3.append(GWcoords2ix(self.world_array, [x, y]))
        # room 4
        for x in range(1 + self.scale + 1, 1 + self.scale + 1 + self.scale):
            for y in range(1 + self.scale + 1, 1 + self.scale + 1 + self.scale):
                self.states_room4.append(GWcoords2ix(self.world_array, [x, y]))
        self.states_room1 = np.array(self.states_room1)
        self.states_room2 = np.array(self.states_room2)
        self.states_room3 = np.array(self.states_room3)
        self.states_room4 = np.array(self.states_room4)
        # bnecks
        holeV = np.ceil(self.scale / 2.0).astype("int")
        holeH = np.ceil(self.scale / 2.0).astype("int")
        self.states_bnecks.append(
            GWcoords2ix(self.world_array, [holeH, 1 + self.scale])
        )  # room 1 to room 2
        self.states_bnecks.append(
            GWcoords2ix(self.world_array, [1 + self.scale, 1 + self.scale + holeV])
        )  #  room 2 to room 4
        self.states_bnecks.append(
            GWcoords2ix(self.world_array, [1 + self.scale, holeV])
        )  #  room 3 to room 4
        self.states_bnecks.append(
            GWcoords2ix(self.world_array, [1 + self.scale + holeH, 1 + self.scale])
        )  # room 1 to room 3
        self.states_bnecks = np.array(self.states_bnecks)

    def _state_information(self):
        """sets information about state including modules/colors etc"""
        # state type indices
        self.state_type_index = {}
        self.state_type_index["bnecks"] = self.states_bnecks
        self.state_type_index["room1"] = self.states_room1
        self.state_type_index["room2"] = self.states_room2
        self.state_type_index["room3"] = self.states_room3
        self.state_type_index["room4"] = self.states_room4

        self.info_state = pd.DataFrame(
            index=pd.Index(range(self.n_state), name="state")
        )
        self.info_state["forced"] = (
            self.A.sum(1) <= 1
        )  # identify states with no or forced policies
        self.info_state["type"] = "Other"
        self.info_state.loc[self.states_bnecks, "type"] = "Bottleneck"
        self.info_state.loc[self.states_room1, "module"] = "Room 1"
        self.info_state.loc[self.states_room2, "module"] = "Room 2"
        self.info_state.loc[self.states_room3, "module"] = "Room 3"
        self.info_state.loc[self.states_room4, "module"] = "Room 4"
        self.info_state["bottleneck_nearest"] = -1
        self.info_state["bottleneck_dist"] = np.nan
        self.info_state["goal_nearest"] = -1
        self.info_state["goal_dist"] = np.nan
        self.info_state["opt_path"] = "Not on optimal path"
        self.info_state["opt_path_pos"] = np.nan
        self.info_state["opt_path_bool"] = False
        self.info_state["label"] = (self.info_state.index + 1).astype(
            "str"
        )  #  label states starting at 1
        self.info_state.loc[self.states_bnecks, "label"] = "+"
        if self.wormhole is not None:
            self.set_info_state(state=self.wormhole[0], type="Wormhole", label="W")
            self.set_info_state(state=self.wormhole[1], type="Wormhole", label="W")
        self.info_state["x"] = np.nan
        self.info_state["y"] = np.nan
        self.info_state = self.info_state.astype(
            dtype={
                "forced": "bool",
                "type": "str",
                "module": "str",
                "goal_nearest": "int",
                "goal_dist": "float",
                "bottleneck_nearest": "int",
                "bottleneck_dist": "float",
                "opt_path": "str",
                "opt_path_pos": "float",
                "opt_path_bool": "bool",
                "x": "float",
                "y": "float",
            }
        )

    def _node_info(self):
        """computes state coordinates from world_array"""
        self.pos = {}
        self.xy = np.zeros((self.n_state, 2))
        for state in self.info_state.index:
            coords = GWix2coords(world_array=self.world_array, ix=state)
            self.set_info_state(state=state, x=coords[0], y=coords[1])
            self.pos[state] = coords
            self.xy[state, :] = coords

    def set_viz_scheme(self, scheme="flat", alphas=[0.3, 1.0]):
        """
        FUNCTION: Sets plotting variables according to info_state
        INPUTS: alphas = transparency values for minor and major plot components respectively
        """
        color_background = "lightgray"
        size_lines = "goal_dist"
        sizes_lines = (3, 0.5)  # inverted for goal_dist
        size_markers = "KL_prior"
        sizes_markers = (300, 200)  # inverted for goal_dist
        dashes = {
            "Bottleneck": (2, 2, 10, 2),
            "Wormhole": (2, 2, 20, 2),
            "Start": "",
            "Goal": "",
            "Switch": "",
            "Other": "",
        }
        markers = {
            "Bottleneck": "P",
            "Wormhole": "*",
            "Start": "o",
            "Goal": "o",
            "Switch": ".",
            "Other": ".",
        }
        style_order = ["Bottleneck", "Wormhole", "Start", "Goal", "Switch", "Other"]
        hue = "state"
        hue_order = None

        self.viz_kwargs = {
            "legend": None,
            "units": "state",
            "hue": hue,
            "hue_order": hue_order,
            "style": "type",
            "style_order": style_order,
            "estimator": None,
        }
        self.viz_kwargs_markers = {
            **self.viz_kwargs,
            "size": size_markers,
            "sizes": sizes_markers,
            "markers": markers,
        }
        self.viz_kwargs_lines = {
            **self.viz_kwargs,
            "size": size_lines,
            "sizes": sizes_lines,
            "dashes": dashes,
        }
        self.state_type_major = [
            "Bottleneck",
            "Wormhole",
            "Start",
            "Goal",
            "Switch",
        ]
        self.state_type_alphas = {
            "minor": alphas[0],
            "major": alphas[1],
        }

        if scheme == "hierarchical":
            # construct palettes
            bnecks = self.state_type_index["bnecks"]
            states_room1 = self.state_type_index["room1"]
            states_room2 = self.state_type_index["room2"]
            states_room3 = self.state_type_index["room3"]
            states_room4 = self.state_type_index["room4"]
            self.color_palette = [
                "#9b59b6",
                "#3498db",
                "#95a5a6",
                "#e74c3c",
                "#34495e",
                "#2ecc71",
            ]
            self.info_state.loc[states_room1, "color_index"] = 3
            self.info_state.loc[states_room2, "color_index"] = 0
            self.info_state.loc[states_room3, "color_index"] = 5
            self.info_state.loc[states_room4, "color_index"] = 1
            self.info_state.loc[bnecks[0], "color_index"] = 3
            self.info_state.loc[bnecks[1], "color_index"] = 0
            self.info_state.loc[bnecks[2], "color_index"] = 5
            self.info_state.loc[bnecks[3], "color_index"] = 1
        elif scheme == "flat":
            self.color_palette = [color_background]
            self.info_state.loc[:, "color_index"] = 0
        else:
            raise ValueError("Unknown color scheme.")
        self.info_state["color"] = self.info_state.color_index.apply(
            lambda x: self.color_palette[int(x)]
        )
        self.info_state = self.info_state.astype(
            dtype={"color_index": "int", "color": "str"}
        )
        self.set_palette()


class GridWorld(GridWorldEnv, GraphEnv):
    """
    Grid world based on Mazelab.
    INPUTS: maze = numpy array indicating available/unavailable states
            start/goal/name/reward/stepcost/goal_absorb/stay_actions as in other envs
    NOTES: takes advantage of MazeLab visualization
    """

    def __init__(
        self,
        world_array,
        start=0,
        goal=5,
        name="gridworld",
        reward=10,
        stepcost=1,
        goal_absorb=False,
        stay_actions=False,
    ):
        self.world_array = world_array
        self._access_matrix()
        self.start = start
        self.goal = goal
        self.maze = Maze(world_array=self.world_array, start=self.start, goal=self.goal)
        GridWorldEnv.__init__(self, maze=self.maze)
        GraphEnv.__init__(self)
        if self.goal is not None:
            self.goal_func(
                reward=reward,
                stepcost=stepcost,
                goals=goal,
                goal_absorb=goal_absorb,
                stay_actions=stay_actions,
            )
        self.n_dim = 2
        self.__type__ = "gridworld"
        self.__name__ = name
        self._node_info()

    def _access_matrix(self):
        self.A = Amat(self.world_array)
        self.A_adj = Amat(self.world_array)
        self.n_state = self.A.shape[0]

    def _node_info(self):
        """computes state coordinates from world_array"""
        self.pos = {}
        self.xy = np.zeros((self.n_state, 2))
        for state in self.info_state.index:
            coords = GWix2coords(world_array=self.world_array, ix=state)
            self.set_info_state(state=state, x=coords[0], y=coords[1])
            self.pos[state] = coords
            self.xy[state, :] = coords

    def draw_graph(self, with_labels=False, ax=None):
        """uses networkx to draw graph"""
        self.stoch_mat_to_trans_weights()
        self._set_graph_from_trans_attr(attr="prob")
        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = plt.gca()

        pos = self._pos_dict()
        edge_weights = nx.get_edge_attributes(self.G, "prob").values()
        edge_weights = [4 * e for e in edge_weights]
        nx.draw(
            self.G,
            pos,
            node_size=80,
            alpha=1,
            node_color="black",
            width=0,
            with_labels=with_labels,
            font_size=24,
            ax=ax,
            arrows=False,
        )
        nx.draw_networkx_edges(
            G=self.G, pos=pos, width=edge_weights, alpha=1, arrows=True, ax=ax
        )
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")


class DecisionTree(GraphEnv):
    """
    Simple decision tree environment.
    Initialize.
    Then set the reward function with reward_func.
    INPUTS: breadth/depth of decision tree
            A = input adjacency array (should be upper triangular for directedness)
            stay_actions = whether the agent can choose to stay at terminal states (or reset)
    NOTES:
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, breadth=2, depth=2, A=None, stay_actions=False, viz_scale=1.0, suffix=""
    ):
        self.__name__ = "decision-tree-b%id%i" % (breadth, depth) + "_" + suffix
        self.__type__ = "graph"
        self.suffix = suffix
        if A is not None:
            self.A = A
            self.breadth_min = None
            self.depth_min = None
            self.breadth_max = None
            self.depth_max = None
        else:
            self.A = None
            self.breadth = breadth
            self.depth = depth
        self.G = None
        self.R = None
        self.xy = None
        self.viz_scale = viz_scale
        self.n_state = None
        self.reward = None
        self.stepcost = None
        self.start = None
        self.goal = None
        self.goal_absorb = False
        self.stay_actions = False
        self.info_state = None
        self.viewer = None

        self._access_matrix()
        self._state_information()
        self._node_info()
        self.set_viz_scheme(
            alphas=[1.0, 1.0], sizes_lines=(2, 2), sizes_markers=(100, 100)
        )

        # action space defined as number of states
        self.action_space = spaces.Discrete(self.n_state)
        self.naction_max = (self.A != 0).sum(axis=1).max()
        # discrete set of states
        self.observation_space = spaces.Discrete(self.n_state)
        # initialize
        self._seed()
        self._reset()

    def _access_matrix(self):
        """Constructs accessibility matrix of decision tree."""
        if self.A is None:
            self.n_state = np.array(
                [self.breadth ** layer for layer in range(self.depth + 1)]
            ).sum()
            self.states = np.arange(self.n_state)
            self.layer = create_tensor((self.n_state,))
            A = create_tensor((self.n_state, self.n_state))
            states_layer = np.array([0])
            self.layer[0] = 0
            for layer in range(1, self.depth + 1):
                states_next_layer = np.arange(
                    states_layer.max() + 1,
                    states_layer.max() + 1 + self.breadth ** layer,
                )
                for ix, i in enumerate(states_layer):
                    for j in states_next_layer[
                        ix * self.breadth : (ix + 1) * self.breadth
                    ]:
                        self.layer[
                            j
                        ] = layer  # layers enumerated from 0 to self.depth-1
                        A[i, j] = 1.0
                states_layer = states_next_layer

            # loop from terminal states to start state
            for i in states_layer:
                A[i, 0] = 1.0
            self.A = A
            self.A_adj = self.A.copy()
            self.G_adj = nx.DiGraph(self.A_adj)
            self._set_graph()
        else:
            self.A_adj = self.A.copy()
            self.G_adj = nx.DiGraph(self.A_adj)
            self.A[self.A.sum(1) == 0, 0] = 1.0
            self._set_graph()
            self.n_state = self.A.shape[0]
            self.states = np.arange(self.n_state)
            self.breadth_min = None
            self.depth_min = None
            self.breadth_max = None
            self.depth_max = None
        self.A_adj = self.A.copy()

    def _node_info(self, method="graphviz"):
        """Defines 2D state node embedding (e.g. for plotting)."""
        if method == "explicit":
            xy = create_tensor((self.n_state, 2))
            x_sep = self.viz_scale / 2.0
            y_sep = self.viz_scale

            # final layer
            layer = self.depth
            x_offsets = x_sep * np.linspace(
                -(self.breadth ** self.depth - 1) / 2.0,
                (self.breadth ** self.depth - 1) / 2.0,
                self.breadth ** self.depth,
            )
            states_layer = self.states[self.layer == layer]
            for ix, i in enumerate(states_layer):
                xy[i, 0] = x_offsets[ix]
                xy[i, 1] = layer * y_sep
            # working backwards
            for layer in reversed(range(self.depth)):
                states_layer = self.states[self.layer == layer]
                states_next_layer = self.states[self.layer == layer + 1]
                for ix, i in enumerate(states_layer):
                    states_succ = states_next_layer[
                        ix * self.breadth : (ix + 1) * self.breadth
                    ]
                    xy[i, 0] = xy[
                        states_succ, 0
                    ].mean()  # average x position of successor states
                    xy[i, 1] = layer * y_sep
        elif method == "graphviz":
            pos = graphviz_layout(self.G_adj, prog="dot")
            xy = create_tensor((self.n_state, 2))
            for key, value in pos.items():
                xy[key, :] = value
            xy[:, 1] = -xy[:, 1]  # y-axis flip
        xy[:, 1] = xy[:, 1] - xy[:, 1].mean()  # offset to center of y-axis
        self.xy = xy
        self.info_state["x"] = self.xy[:, 0]
        self.info_state["y"] = self.xy[:, 1]
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]


class LiuSequences(GridWorld):
    def __init__(self, suffix=""):
        world_array = np.ones((4, 7))
        world_array[1, 1:6] = 0  # seq 1
        world_array[2, 1:6] = 0  # seq 2
        super(LiuSequences, self).__init__(world_array=world_array)
        self.suffix = suffix
        self.__name__ = "liu-sequences" + "_" + suffix
        self.G = nx.DiGraph(self.G)

    def _state_information(self):
        """sets information about state including modules/colors etc"""
        # state type indices
        self.states_seq1 = [0, 1, 2, 3]
        self.states_seq2 = [5, 6, 7, 8]
        self.states_reward = [4, 9]

        self.state_type_index = {}
        self.state_type_index["seq1"] = self.states_seq1
        self.state_type_index["seq2"] = self.states_seq2
        self.state_type_index["reward"] = self.states_reward

        self.info_state = pd.DataFrame(
            index=pd.Index(range(self.n_state), name="state")
        )
        self.info_state.loc[self.states_reward, "type"] = "reward"
        self.info_state.loc[self.states_seq1, "type"] = "seq"
        self.info_state.loc[self.states_seq2, "type"] = "seq"
        self.info_state.loc[self.states_seq1, "sequence"] = "1"
        self.info_state.loc[self.states_seq2, "sequence"] = "2"
        self.info_state.loc[:, "label"] = (self.info_state.index + 1).astype(
            "str"
        )  #  label states starting at 1

        self.info_state.loc[:, "x"] = np.nan
        self.info_state.loc[:, "y"] = np.nan
        self.info_state = self.info_state.astype(
            dtype={
                "type": "str",
                "sequence": "str",
                "label": "str",
                "x": "float",
                "y": "float",
            }
        )

    def set_viz_scheme(self):
        """
        FUNCTION: Sets plotting variables according to viz_scheme and info_state
        INPUTS: viz_scheme = colors/sizes/dashes etc scheme
                alphas = transparency values for minor and major plot components respectively
        """
        # construct palettes
        self.info_state.loc[self.states_seq1, "color_index"] = 0
        self.info_state.loc[self.states_seq2, "color_index"] = 1
        self.info_state.loc[self.states_reward, "color_index"] = 2
        self.color_palette = sb.hls_palette(3, h=0.01, l=0.5, s=0.65).as_hex()
        self.info_state["color"] = self.info_state.color_index.apply(
            lambda x: self.color_palette[int(x)]
        )
        self.info_state = self.info_state.astype(
            dtype={"color_index": "int", "color": "str"}
        )
        # self.set_palette()

    def draw_graph(self, drop_reward=True, with_labels=False, ax=None):
        """uses networkx to draw graph"""
        G = self.G.copy(as_view=False)
        if drop_reward:
            G.remove_node(self.states_reward[0])
            G.remove_node(self.states_reward[1])

        # eliminate inter-sequence arrows
        for i, x in enumerate(self.states_seq1):
            y = self.states_seq2[i]
            G.remove_edge(x, y)
            G.remove_edge(y, x)
        # eliminate backward-sequence arrows
        for i in range(len(self.states_seq1) - 1):
            x1 = self.states_seq1[i]
            y1 = self.states_seq1[i + 1]
            x2 = self.states_seq2[i]
            y2 = self.states_seq2[i + 1]
            G.remove_edge(y1, x1)
            G.remove_edge(y2, x2)

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        pos = self._pos_dict()
        nx.draw(
            G,
            pos,
            node_size=50,
            alpha=0.7,
            node_color="black",
            width=0,
            with_labels=with_labels,
            font_size=24,
            ax=ax,
            arrows=False,
        )
        nx.draw_networkx_edges(
            G=G, pos=pos, width=1.0, alpha=0.6, arrows=True, ax=ax
        )
        plt.axis("equal")
        plt.axis("off")

    def plot_environment(self, X=None, ax=None, figsize=(12, 12)):
        """
        FUNCTION: plot environment graph
        """
        self._set_graph(X=X)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.axes(ax)
        self.draw_graph(ax=ax)


class TowerofHanoi(GraphEnv):
    def __init__(self, viz_scheme="colors_task", viz_scale=1.0, suffix=""):
        super(GraphEnv, self).__init__()
        self.__name__ = "tower-of-hanoi" + "_" + suffix
        self.__type__ = "graph"
        self.suffix = suffix
        self.A = None
        self.G = None
        self.R = None
        self.xy = None
        self.n_state = 27
        self.reward = None
        self.stepcost = None
        self.start = None
        self.goal = None
        self.info_state = None
        self.viz_scale = viz_scale

        self._access_matrix()
        self._state_information()
        self._node_info()
        self.set_viz_scheme(viz_scheme=viz_scheme)
        self.goal_func()

    def set_palette(self, var=None):
        """Copies info_state.color information to palette grouped by var."""
        if var is None:
            var = "hue"
        self.viz_kwargs["palette"] = (
            self.info_state.groupby(self.viz_kwargs[var])
            .apply(lambda x: x.loc[x.index[0], "color"])
            .to_dict()
        )
        self.viz_kwargs_lines["palette"] = self.viz_kwargs["palette"]
        self.viz_kwargs_markers["palette"] = self.viz_kwargs["palette"]

    def set_viz_scheme(self, viz_scheme="colors_task", alphas=[0.3, 1.0]):
        """
        FUNCTION: Sets plotting variables according to viz_scheme and info_state
        INPUTS: viz_scheme = colors/sizes/dashes etc scheme
                alphas = transparency values for minor and major plot components respectively
        """
        self.viz_scheme = viz_scheme
        if self.viz_scheme == "colors_task":
            size_lines = "goal_dist"
            sizes_lines = (3, 0.5)  # inverted for goal_dist
            size_markers = "KL_prior"
            sizes_markers = (300, 200)  # inverted for goal_dist
            dashes = {
                "Bottleneck": (2, 2, 10, 2),
                "Start": "",
                "Goal": "",
                "Switch": "",
                "Other": "",
            }
            markers = {
                "Bottleneck": "P",
                "Start": "o",
                "Goal": "o",
                "Switch": "o",
                "Other": ".",
            }
            style_order = ["Bottleneck", "Start", "Goal", "Switch", "Other"]
            # state based
            hue = "state"
            hue_order = None
        elif self.viz_scheme == "colors_hierarchy":
            size_lines = None
            size_markers = None
            sizes_lines = None
            sizes_markers = None
            dashes = {"Bottleneck": (2, 2, 10, 2), "Start": "", "Goal": "", "Other": ""}
            markers = {
                "Bottleneck": "P",
                "Start": "o",
                "Goal": "o",
                "Other": ".",
            }
            style_order = ["Bottleneck", "Start", "Goal", "Switch", "Other"]
            hue = "state"
            hue_order = None
        else:
            raise ValueError("Unknown viz scheme.")
        self.viz_kwargs = {
            "legend": None,
            "units": "state",
            "hue": hue,
            "hue_order": hue_order,
            "style": "type",
            "style_order": style_order,
            "estimator": None,
        }
        self.viz_kwargs_markers = {
            **self.viz_kwargs,
            "size": size_markers,
            "sizes": sizes_markers,
            "markers": markers,
        }
        self.viz_kwargs_lines = {
            **self.viz_kwargs,
            "size": size_lines,
            "sizes": sizes_lines,
            "dashes": dashes,
        }
        self.state_type_major = [
            "Bottleneck",
            "Start",
            "Goal",
            "Switch",
        ]
        self.state_type_alphas = {
            "minor": alphas[0],
            "major": alphas[1],
        }

        # construct palettes
        bnecks = self.state_type_index["bnecks"]
        states_mod1 = self.state_type_index["states_mod1"]
        states_mod2 = self.state_type_index["states_mod2"]
        states_mod3 = self.state_type_index["states_mod3"]
        bnecks1 = self.state_type_index["bnecks1"]
        bnecks2 = self.state_type_index["bnecks2"]
        bnecks3 = self.state_type_index["bnecks3"]
        if self.viz_scheme == "colors_hierarchy":
            self.color_palette = sb.hls_palette(6, h=0.01, l=0.5, s=0.65).as_hex()
            self.info_state.loc[states_mod1, "color_index"] = 0
            self.info_state.loc[states_mod2, "color_index"] = 2
            self.info_state.loc[states_mod3, "color_index"] = 4
            self.info_state.loc[bnecks1, "color_index"] = 1
            self.info_state.loc[bnecks2, "color_index"] = 5
            self.info_state.loc[bnecks3, "color_index"] = 3
            self.info_state["color"] = self.info_state.color_index.apply(
                lambda x: self.color_palette[int(x)]
            )
        elif self.viz_scheme == "colors_task":
            self.color_palette = [
                "#9b59b6",
                "#3498db",
                "#95a5a6",
                "#e74c3c",
                "#34495e",
                "#2ecc71",
            ]
            self.info_state.loc[states_mod1, "color_index"] = 0
            self.info_state.loc[states_mod2, "color_index"] = 2
            self.info_state.loc[states_mod3, "color_index"] = 4
            self.info_state["color"] = self.info_state.color_index.apply(
                lambda x: self.color_palette[int(x)]
            )
            self.info_state.loc[bnecks1, "label"] = "+"  #  'B'
            self.info_state.loc[bnecks2, "label"] = "+"
            self.info_state.loc[bnecks3, "label"] = "+"
        self.info_state = self.info_state.astype(
            dtype={"color_index": "int", "color": "str"}
        )
        self.set_palette()
        if hasattr(self, "goals"):
            self._info_goal()

    def _access_matrix(self):
        self.n_state = 27
        self.A = create_tensor((self.n_state, self.n_state))

        # only encode transitions "down" and "right" and then symmetrize
        self.A[0, [1, 2]] = 1
        self.A[1, [2, 3]] = 1
        self.A[2, [4]] = 1
        self.A[3, [5, 6]] = 1
        self.A[4, [7, 8]] = 1
        self.A[5, [6, 9]] = 1
        self.A[6, [7]] = 1
        self.A[7, [8]] = 1
        self.A[8, [10]] = 1
        self.A[9, [11, 12]] = 1
        self.A[10, [13, 14]] = 1
        self.A[11, [12, 15]] = 1
        self.A[12, [16]] = 1
        self.A[13, [14, 17]] = 1
        self.A[14, [18]] = 1
        self.A[15, [19, 20]] = 1
        self.A[16, [21, 22]] = 1
        self.A[17, [23, 24]] = 1
        self.A[18, [25, 26]] = 1
        self.A[19, [20]] = 1
        self.A[20, [21]] = 1
        self.A[21, [22]] = 1
        self.A[22, [23]] = 1
        self.A[23, [24]] = 1
        self.A[24, [25]] = 1
        self.A[25, [26]] = 1

        # symmetrize
        self.A = np.triu(self.A) + np.triu(self.A, 1).T
        self.A_adj = self.A.copy()

    def _node_info(self):
        self.xy = np.zeros((self.n_state, 2))
        xstep = self.viz_scale / 2.0
        ystep = self.viz_scale
        yoffset = (8 - 1) * self.viz_scale / 2.0
        # layer 1
        self.xy[0, :] = [0.0, 0.0]
        # layer 2
        self.xy[1, :] = [self.xy[0, 0] - xstep, self.xy[0, 1] - ystep]
        self.xy[2, :] = [self.xy[0, 0] + xstep, self.xy[0, 1] - ystep]
        # layer 3
        self.xy[3, :] = [self.xy[1, 0] - xstep, self.xy[1, 1] - ystep]
        self.xy[4, :] = [self.xy[2, 0] + xstep, self.xy[2, 1] - ystep]
        # layer 4
        self.xy[5, :] = [self.xy[3, 0] - xstep, self.xy[3, 1] - ystep]
        self.xy[6, :] = [self.xy[3, 0] + xstep, self.xy[3, 1] - ystep]
        self.xy[7, :] = [self.xy[4, 0] - xstep, self.xy[4, 1] - ystep]
        self.xy[8, :] = [self.xy[4, 0] + xstep, self.xy[4, 1] - ystep]
        # layer 5
        self.xy[9, :] = [self.xy[5, 0] - xstep, self.xy[5, 1] - ystep]
        self.xy[10, :] = [self.xy[8, 0] + xstep, self.xy[8, 1] - ystep]
        # layer 6
        self.xy[11, :] = [self.xy[9, 0] - xstep, self.xy[9, 1] - ystep]
        self.xy[12, :] = [self.xy[9, 0] + xstep, self.xy[9, 1] - ystep]
        self.xy[13, :] = [self.xy[10, 0] - xstep, self.xy[10, 1] - ystep]
        self.xy[14, :] = [self.xy[10, 0] + xstep, self.xy[10, 1] - ystep]
        # layer 7
        self.xy[15, :] = [self.xy[11, 0] - xstep, self.xy[11, 1] - ystep]
        self.xy[16, :] = [self.xy[12, 0] + xstep, self.xy[12, 1] - ystep]
        self.xy[17, :] = [self.xy[13, 0] - xstep, self.xy[13, 1] - ystep]
        self.xy[18, :] = [self.xy[14, 0] + xstep, self.xy[14, 1] - ystep]
        # layer 8
        self.xy[19, :] = [self.xy[15, 0] - xstep, self.xy[15, 1] - ystep]
        self.xy[20, :] = [self.xy[15, 0] + xstep, self.xy[15, 1] - ystep]
        self.xy[21, :] = [self.xy[16, 0] - xstep, self.xy[16, 1] - ystep]
        self.xy[22, :] = [self.xy[16, 0] + xstep, self.xy[16, 1] - ystep]
        self.xy[23, :] = [self.xy[17, 0] - xstep, self.xy[17, 1] - ystep]
        self.xy[24, :] = [self.xy[17, 0] + xstep, self.xy[17, 1] - ystep]
        self.xy[25, :] = [self.xy[18, 0] - xstep, self.xy[18, 1] - ystep]
        self.xy[26, :] = [self.xy[18, 0] + xstep, self.xy[18, 1] - ystep]

        self.info_state["x"] = self.xy[:, 0]
        self.info_state["y"] = self.xy[:, 1]
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    def _state_information(self):
        """sets information about state including modules/colors etc"""
        # state type indices
        bnecks = [5, 8, 9, 10, 22, 23]
        states_mod1 = list(range(9))
        states_mod2 = [9, 11, 12, 15, 16, 19, 20, 21, 22]
        states_mod3 = [10, 13, 14, 17, 18, 23, 24, 25, 26]
        bnecks1 = [5, 9]
        bnecks2 = [8, 10]
        bnecks3 = [22, 23]
        self.state_type_index = {}
        self.state_type_index["bnecks"] = bnecks
        self.state_type_index["states_mod1"] = states_mod1
        self.state_type_index["states_mod2"] = states_mod2
        self.state_type_index["states_mod3"] = states_mod3
        self.state_type_index["bnecks1"] = bnecks1
        self.state_type_index["bnecks2"] = bnecks2
        self.state_type_index["bnecks3"] = bnecks3

        self.info_state = pd.DataFrame(
            index=pd.Index(range(self.n_state), name="state")
        )
        self.info_state["forced"] = (
            self.A.sum(1) <= 1
        )
        self.info_state["type"] = "Other"
        self.info_state.loc[bnecks, "type"] = "Bottleneck"
        self.info_state.loc[states_mod1, "module"] = "Module 1"
        self.info_state.loc[states_mod2, "module"] = "Module 2"
        self.info_state.loc[states_mod3, "module"] = "Module 3"
        self.info_state["bottleneck_nearest"] = -1
        self.info_state["bottleneck_dist"] = np.nan
        self.info_state["goal_nearest"] = -1
        self.info_state["goal_dist"] = np.nan
        self.info_state["opt_path"] = "Not on optimal path"
        self.info_state["opt_path_pos"] = np.nan
        self.info_state["opt_path_bool"] = False
        self.info_state["label"] = (self.info_state.index + 1).astype(
            "str"
        )

        self.__bottleneck_distance()
        self.info_state["x"] = np.nan
        self.info_state["y"] = np.nan
        self.info_state = self.info_state.astype(
            dtype={
                "forced": "bool",
                "type": "str",
                "module": "str",
                "goal_nearest": "int",
                "goal_dist": "float",
                "bottleneck_nearest": "int",
                "bottleneck_dist": "float",
                "opt_path": "str",
                "opt_path_pos": "float",
                "opt_path_bool": "bool",
                "x": "float",
                "y": "float",
            }
        )

    def goal_func(
        self,
        reward=10,
        stepcost=1,
        goals=3,
        goal_absorb=False,
        stay_actions=True,
        goal_remain=False,
    ):
        """
            FUNCTION: Defines reward function.
            INPUTS: reward = reward for transition to a goal state
                    stepcost = cost for transitioning to a non-goal state
                    goals = goal states (can choose several)
                    goal_absorb = absorb at goals (not yet implemented in PPP)
                    stay_actions = False implies dissipative MDP without option to stay at states
                                True implies dissipative MDP with option to stay at states
                    goal_remain = forced to stay at goal state
                    goal_absorb/stay_actions/goal_remain = False, default is to restart episode (wormhole from goal to start)
        """
        if not hasattr(goals, "__len__"):
            goals = [goals]
        assert np.all(np.array(goals) < self.n_state)
        self.stay_actions = stay_actions
        self.goal_absorb = goal_absorb
        self.goal_remain = goal_remain
        self.reward = reward
        self.stepcost = stepcost
        self.goals = goals

        # modify accessibility matrix self.A
        if self.goal_absorb:
            # make symmetric adjacency copy of non-symmetric accessibility matrix self.A
            self.A_adj = self.A.copy()
        for g in self.goals:
            if self.goal_absorb:
                self.A[g, :] = 0  # no edges available from goal state (thus absorbing)
            elif self.stay_actions:
                # option to stay at any state (as opposed to reset)
                np.fill_diagonal(self.A, 1)
            elif self.goal_remain:
                self.A[g, :] = 0
                self.A[g, g] = 1
            else:
                #  dissipative MDP (reset to start)
                self.A[g, :] = 0
                self.A[g, self.start] = 1
            self._set_state_forced()
            self.info_state.loc[g, "opt_path"] = "Goal"
            self.info_state.loc[g, "label"] = "G"
            self.info_state.loc[g, "type"] = "Goal"
        # reward function
        self.R = create_tensor((self.n_state, self.n_state), fill_value=-np.inf)
        for i in range(self.n_state):
            for j in range(self.n_state):
                if self.A[i, j] != 0:
                    if j in self.goals:
                        self.R[i, j] = reward
                    else:
                        self.R[i, j] = -stepcost
        if not self.goal_absorb and not self.stay_actions and not self.goal_remain:
            # no restart cost if goal resetting
            for goal in self.goals:
                self.R[goal, self.start] = 0
        self._compute_reward_per_state()
        self.info_state.loc[self.start, "opt_path"] = "Start"
        self.info_state.loc[self.start, "label"] = "S"
        self.info_state.loc[self.start, "type"] = "Start"
        self._label_shortest_paths()
        self._info_goal()

    def _label_shortest_paths(self):
        """
        FUNCTION: Identifies shortest path from start to goal.
        NOTES: For multiple goals, solves for the nearest goal.
               For multiple shortest paths of equal length, records them all.
        """
        import networkx as nx

        self._set_graph()
        if len(self.goals) > 1:
            # find goal with shortest path length
            current_shortest_length = np.inf
            for g in self.goals:
                goal_shortest_length = nx.shortest_path_length(
                    self.G, source=self.start, target=g
                )
                if goal_shortest_length < current_shortest_length:
                    current_shortest_length = goal_shortest_length
                    goal_shortest = g
        else:
            goal_shortest = self.goals[0]
        paths = list(
            nx.all_shortest_paths(self.G, source=self.start, target=goal_shortest)
        )
        for path in paths:
            for ix, state in enumerate(path):
                self.info_state.loc[state, "opt_path_bool"] = True
                if self.info_state.loc[state, "opt_path"] not in ["Start", "Goal"]:
                    self.info_state.loc[state, "opt_path"] = "Via state"
                self.info_state.loc[state, "opt_path_pos"] = ix

    def __bottleneck_distance(self):
        """
        Record nearest bottleneck to a state as well as the distance to that bottleneck
        """
        self._set_graph()
        bottlenecks = self.info_state[self.info_state.type == "Bottleneck"].index
        for state in range(self.n_state):
            shortest_lengths = nx.shortest_path_length(self.G, source=state)
            bneck_shortest_lengths = np.array(
                [shortest_lengths[i] for i in bottlenecks]
            )
            self.info_state.loc[state, "bottleneck_nearest"] = bottlenecks[
                bneck_shortest_lengths.argmin()
            ]
            self.info_state.loc[state, "bottleneck_dist"] = bneck_shortest_lengths.min()

    def _info_goal(self):
        """
        Record nearest goal to a state as well as the distance to that goal.
        Adapts state color scheme to reflect task-orientation in "colors_task" scheme.
        """
        import networkx as nx

        self._set_graph()
        goals = self.goals
        for state in range(self.n_state):
            shortest_lengths = nx.shortest_path_length(self.G, source=state)
            goal_shortest_lengths = np.array([shortest_lengths[i] for i in goals])
            self.info_state.loc[state, "goal_nearest"] = goals[
                goal_shortest_lengths.argmin()
            ]
            self.info_state.loc[state, "goal_dist"] = goal_shortest_lengths.min()

        # reflect task in module labels
        self.info_state.replace(
            {self.info_state.module[self.start]: "Start module"}, inplace=True
        )
        # color goal module for first goal in list
        g = self.goals[0]
        self.info_state.replace(
            {self.info_state.module[g]: "Goal module"}, inplace=True
        )
        if not (np.array(self.goals) == self.goals[0]).all():
            print(
                "Goals distributed over multiple modules, suggest use colors_hierarchy viz_scheme"
            )
        # rename other modules
        modules_other = [
            i
            for i in self.info_state.module.unique()
            if i not in ["Start module", "Goal module"]
        ]
        for module in modules_other:
            self.info_state.replace({module: "Other module"}, inplace=True)

        if self.viz_scheme == "colors_task":
            # overwrites colors_task viz_scheme with task-relevant information
            self.info_state.loc[
                self.info_state.module == "Start module", "color_index"
            ] = 3
            self.info_state.loc[
                self.info_state.module == "Goal module", "color_index"
            ] = 1
            self.info_state.loc[
                self.info_state.module == "Other module", "color_index"
            ] = 2
            self.info_state["color"] = self.info_state.color_index.apply(
                lambda x: self.color_palette[int(x)]
            )
            self.set_palette()


class CircularTrack(GraphEnv):
    def __init__(
        self,
        n_state=27,
        reward=10,
        stepcost=1,
        start=1,
        goal_no=1,
        goal_weight=20.,
        goal_absorb=True,
        backmove=False,
        opt_policy_weight=0.9
    ):
        """
            Circular track environment with/without "goal rooms" (as in Wikenheiser2015)
            INPUTS: n_state: number of states in track
                    reward = reward value
                    stepcost = cost per step taken
                    goal_no = 1,2,3, which goal in Wikenheiser task, otherwise None
                    goal_weight <-> opt_policy_weight, see task_wikenheiser2015 member function
        """
        self.n_state = n_state
        self.reward = reward
        self.stepcost = stepcost
        self.start = start
        self.goal_no = goal_no
        self.goal_weight = goal_weight
        self.goal_absorb = goal_absorb
        self.backmove = backmove
        self.opt_policy_weight = opt_policy_weight
        if goal_no == None:
            self.wikenheiser2015 = False
        else:
            self.wikenheiser2015 = True
        self._access_matrix(goal_rooms=self.wikenheiser2015)
        super(GraphEnv, self).__init__()
        self._state_information()
        self._transition_information()
        self._node_info()
        self.__name__ = "circular-track"
        self.__type__ = "graphworld"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/circular_track.png"
        self._set_graph()
        self.n_dim = 2
        if self.wikenheiser2015:
            self.task_wikenheiser2015(
                goal_no=goal_no, goal_weight=goal_weight, goal_absorb=goal_absorb, backmove=backmove, opt_policy_weight=opt_policy_weight
            )

    def _access_matrix(self, goal_rooms=False, complete_circle=True):
        """Get the adjacency + xy locations for a circular track.

        Args:
          goal_rooms = True implies that states are off the main track
                       mimicking Wikenheiser2015 environment.
          complete_circle = connect last state to first so state-space is topological circle
        Sets:
          A: adjacency matrix
        NOTE: new version, goal states intermixed within state list
        """
        self.goal_rooms = goal_rooms
        if goal_rooms:
            assert self.n_state % 3 == 0, "n_states should be divisible by 3"
            self.n_goal_state = 3
        else:
            self.n_goal_state = 0
        self.n_state_via = self.n_state - self.n_goal_state
        self.last_circle_state = (self.n_state - 1) - 1 # last state on main circular track

        # create circle
        adjmat_upper_tri = np.concatenate(
            [
                np.concatenate(
                    [np.zeros((self.n_state_via - 1, 1)), np.eye(self.n_state_via - 1)],
                    axis=1,
                ),
                np.zeros((1, self.n_state_via)),
            ],
            axis=0,
        )
        if complete_circle:
            adjmat_upper_tri[self.n_state_via - 1, 0] = 1 # last state on track to first

        if goal_rooms:
            # add goal room states at intervals of n_state/3
            self.goal_via_step = int(self.n_state / 3)
            self.goal_states = [
                self.goal_via_step - 1,
                2 * self.goal_via_step - 1,
                3 * self.goal_via_step - 1,
            ]
            self.goal_vias = [g - 1 for g in self.goal_states]
            self.via_states = [
                s for s in range(self.n_state) if s not in self.goal_states
            ]
            A_orig = adjmat_upper_tri.copy()
            adjmat_upper_tri = np.zeros((self.n_state, self.n_state))
            adjmat_upper_tri[np.ix_(self.via_states, self.via_states)] = A_orig
            # glue goal states/rooms to circular track
            adjmat_upper_tri[self.goal_vias[0], self.goal_states[0]] = 1.0
            adjmat_upper_tri[self.goal_vias[1], self.goal_states[1]] = 1.0
            adjmat_upper_tri[self.goal_vias[2], self.goal_states[2]] = 1.0
        self.A = adjmat_upper_tri + adjmat_upper_tri.T
        self.A_adj = self.A.copy()  # backup of underlying adjacency structure


    def task_wikenheiser2015(self, goal_no=1, goal_weight=20., goal_absorb=True, backmove=False, opt_policy_weight=0.9):
        """
        FUNCTION: Sets up environment to reflect Wikenheiser2015 task
        INPUTS: goal_no             = 1,2,3, which goal is active
                goal_weight         = weight on goal-directed transitions in policy
                goal_absorb         = absorb at goal
                backmove            = allow rodent to move backwards around track if passed only goal
                opt_policy_weight   = weight to combine the optimal policy with diffusion generator
                access matrix A modified to (1-w)A + wPI where PI is the optimal policy transition matrix
        NOTE: new version, goal states intermixed within state list
        """
        assert 0<opt_policy_weight<=1, 'policy weight should be between 0 and 1'
        self.opt_policy_weight = opt_policy_weight
        self.PI = np.zeros(self.A.shape)
        if type(goal_no) == int:
            self.goal_via = self.goal_vias[goal_no - 1]
            self.goal_state = self.goal_states[goal_no - 1]
            self.goal_alts = [g for g in self.goal_states if g != self.goal_state] # inactive, alternative goals

            if backmove is False:
                # one-way all way around circular track
                # setting this leads to a random walk past the goal state
                # i.e. assume rodent random walks with goal access until experimenter interferes
                for i,v in enumerate(self.via_states[:-1]):
                    v_next = self.via_states[i+1]
                    self.PI[v,v_next] = 1.
            else:
                # from track states towards goal
                for i,v in enumerate(self.via_states[:-1]):
                    if v < self.goal_state-1:
                        v_next = self.via_states[i+1]
                        self.PI[v,v_next] = 1.
            # from track states BACK towards start along circle
            for i, v1 in enumerate(self.via_states[1:]):
                if v1 > self.goal_state:
                    v2 = self.via_states[i] # previous (wrt anti-clockwise) via state
                    self.PI[v1, v2] = 1.

            # goal turn-off
            self.PI[self.goal_via, self.goal_state] = goal_weight
            # from alternate goal states to circle
            for a in self.goal_alts:
                self.PI[a, a - 1] = 1.0

            if goal_absorb:
                self.PI[self.goal_state, self.goal_state] = goal_weight # absorb at goal
        elif goal_no == 'all':
            # policy optimized for "all" goals
            # counterclockwise around environment
            for i in range(self.n_state_via - 1):
                v1 = self.via_states[i]
                v2 = self.via_states[i+1]
                self.PI[v1,v2] = 1.0
            # turn offs for goals
            for gi in range(3):
                v = self.goal_vias[gi]
                g = self.goal_states[gi]
                self.PI[v, g] = goal_weight
                if goal_absorb:
                    self.PI[g, g] = goal_weight  #  absorb
        else:
            raise ValueError('mis-specified goal')
        # normalize and process stochastic matrix
        self.T = (1 - opt_policy_weight) * self.A + opt_policy_weight * self.PI
        self.T[0, self.last_circle_state] = 0.0 # dont go backwards from start
        self.T[self.last_circle_state, 0] = 0.0 # no return, episode ends at goals
        self.T = row_norm(self.T)
        self.stoch_mat_to_trans_weights()
        self._set_graph(attr="prob")
        self.Gstoch = self.G.copy()

    def _node_info(self, radius=1.0):
        """
        FUNCTION: Defines node positions for plotting.
        INPUTS: radius = graph node position radius
        NOTE: new version, goal states intermixed within state list
        """
        # note 45 degree rotation to match Wikenheiser2015 plot
        self.angles = (
            np.linspace(0, 2 * np.pi, self.n_state + 1)[: self.n_state]
            + np.pi * 45 / 180.0
        )
        self.angles = (
            self.angles + 2 * np.pi / self.n_state
        )  # rotate by one state since the initial state is 1
        self.radius = radius
        self.radius_ext = (
            2 * np.pi * self.radius / self.n_state
        )  # ensures consistent node position distance
        self.radii = np.ones(self.n_state) * radius
        if self.goal_rooms:
            # add offset positions for goal rooms
            self.angles = (
                np.linspace(0, 2 * np.pi, self.n_state_via + 1)[: self.n_state_via]
                + np.pi * 45 / 180.0
            )
            for g in self.goal_states:
                self.angles = np.insert(self.angles, g, self.angles[g - 1])
            self.radii[self.goal_states] = self.radii[self.goal_states] * (
                1.0 + self.radius_ext
            )
        x, y = pol2cart(rho=self.radii, phi=self.angles)
        xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        self.xy = xy
        self.info_state.loc[:, "x"] = x
        self.info_state.loc[:, "y"] = y
        self.info_state.loc[:, "rho"] = self.radii
        self.info_state.loc[:, "phi"] = self.angles
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]
        self._transition_information()

    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self, attr="weight"):
        """Defines networkx graph including info_state information"""
        # extract node/edge attributes
        self._set_graph_from_trans_attr(attr=attr)
        nodesdf = self.info_state.reset_index()
        nx.set_node_attributes(self.G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(self.G, name="y", values=nodesdf.y.to_dict())
        nx.set_node_attributes(self.G, name="rho", values=nodesdf.rho.to_dict())
        nx.set_node_attributes(self.G, name="phi", values=nodesdf.phi.to_dict())

    def distance(self, state1, state2, interval_size=1.0):
        """distance between state1 and state2 = shortest_path_length(A) x interval_size"""
        if not hasattr(self, "shortest_n_steps"):
            self.shortest_n_steps = dict(nx.shortest_path_length(self.G))
        return self.shortest_n_steps[state1][state2] * interval_size

    def _set_graph_from_trans_attr(self, attr="prob"):
        edgesdf = self.info_transition
        self.G = nx.from_pandas_edgelist(
            df=edgesdf,
            source="source",
            target="target",
            edge_attr=attr,
            create_using=nx.DiGraph(),
        )
        remove_list = [
            edge for edge in self.G.edges() if self.G.edges[edge[0], edge[1]][attr] == 0
        ]
        self.G.remove_edges_from(remove_list)

    def plot_environment(self, ax=None, figsize=(12, 12), fname="circular_track.png"):
        """
        FUNCTION: Circular track plot inspired by Wikenheiser & Redish (2015)
        REF: Figure 2, Wikenheiser & Redish (2015)
        """
        self._set_graph()
        if ax is None:
            fig, axes = plt.subplots(figsize=figsize)
        else:
            plt.axes(ax)
        # settings
        circle_lw = 80
        color = "lightgrey"
        circle_radius = self.G.nodes[0]["rho"]
        rect_height = (2 * self.radius_ext) * circle_radius
        rect_width = (2 * self.radius_ext) * circle_radius

        circle = plt.Circle(
            xy=(0, 0),
            radius=circle_radius,
            color=color,
            fill=False,
            linewidth=circle_lw,
        )
        phi1 = np.pi * 30 / 180.0
        phi2 = np.pi * 150 / 180.0
        phi3 = np.pi * 270 / 180.0
        angle1 = -60
        angle2 = 60
        angle3 = 0
        # transform center points to left-bottom point in polar coordinates
        rad1 = circle_radius * 0.99
        rad2 = circle_radius * 0.99
        rad3 = (circle_radius + rect_height) * 0.99
        phi1 = phi1 + np.arctan(rect_width / (2 * rad1))
        phi2 = phi2 + np.arctan(rect_width / (2 * rad2))
        phi3 = phi3 - np.arctan(rect_width / (2 * rad3))
        x1, y1 = pol2cart(rad1, phi1)  # top-right
        x2, y2 = pol2cart(rad2, phi2)  # top-left
        x3, y3 = pol2cart(rad3, phi3)  # bottom
        rect1 = plt.Rectangle(
            xy=(x1, y1),
            width=rect_width,
            height=rect_height,
            angle=angle1,
            color=color,
            fill=True,
        )
        rect2 = plt.Rectangle(
            xy=(x2, y2),
            width=rect_width,
            height=rect_height,
            angle=angle2,
            color=color,
            fill=True,
        )
        rect3 = plt.Rectangle(
            xy=(x3, y3),
            width=rect_width,
            height=rect_height,
            angle=angle3,
            color=color,
            fill=True,
        )
        plt.gca().add_patch(circle)
        if self.wikenheiser2015:
            plt.gca().add_patch(rect1)
            plt.gca().add_patch(rect2)
            plt.gca().add_patch(rect3)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

        if fname is not None:
            plt.savefig("figures/" + fname)
        return plt.gca()


class ConnectedCaveman(GraphEnv):
    def __init__(self, n_clique=3, n_state_clique=5, p=0.0, start=0):
        """
        Connected caveman graph environment (generalizes Schapiro2013)
        NOTES: No goals/rewards/etc since experiment was purely passive/unsupervised rep learning
        https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.connected_caveman_graph.html#networkx.generators.community.connected_caveman_graph
        """

        self.start = start
        self._access_matrix(n_clique, n_state_clique, p)
        super(ConnectedCaveman, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "caveman-graph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/caveman.png"

    def _access_matrix(self, n_clique=3, n_state_clique=5, p=0.0):
        """
        Sets the adjacency/stochastic matrix for the connected caveman graph.
        INPUTS:  n_clique = number of cliques
                 n_state_clique = number of states per clique
                 p = noisy connection probability
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        """
        self.n_clique = n_clique
        self.n_state_clique = n_state_clique
        self.n_state = n_clique * n_state_clique
        self.p = p
        if self.p != 0:
            from networkx.generators.community import relaxed_caveman_graph

            self.G = relaxed_caveman_graph(n_clique, n_state_clique, p)
        else:
            from networkx.generators.community import connected_caveman_graph

            self.G = connected_caveman_graph(n_clique, n_state_clique)
        self.A = nx.adjacency_matrix(self.G).toarray()
        self.A_adj = self.A
        self.T = row_norm(self.A)
        self._check_statespace()

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions.
        NOTES: https://networkx.github.io/documentation/networkx-1.9/reference/drawing.html
        """
        self.layout = nx.spring_layout(
            self.G,
            dim=2,
            k=None,
            pos=None,
            fixed=None,
            iterations=50,
            weight="weight",
            scale=1.0,
        )
        # self.layout = nx.shell_layout(self.G)
        self.info_state.loc[self.layout.keys(), "x"] = [
            self.layout[i][0] for i in self.layout.keys()
        ]
        self.info_state.loc[self.layout.keys(), "y"] = [
            self.layout[i][1] for i in self.layout.keys()
        ]
        self.xy = np.zeros((self.n_state, 2))
        self.xy[:, 0] = self.info_state.loc[:, "x"]
        self.xy[:, 1] = self.info_state.loc[:, "y"]
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    def draw_graph(
        self,
        node_size=100,
        edge_width=1,
        with_labels=False,
        color_communities=False,
        ax=None,
    ):
        """uses networkx to draw graph"""
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca()

        pos = self._pos_dict()
        nx.draw_networkx_nodes(
            self.G.to_undirected(),
            pos,
            node_size=node_size,
            linewidths=0,
            edgecolors="black",
            node_color="black",
            with_labels=with_labels,
            font_size=24,
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.G.to_undirected(), pos, edge_color="k", width=edge_width, alpha=0.5
        )
        plt.tight_layout()
        ax.axis("equal")
        ax.axis("off")


class RingOfCliques(GraphEnv):
    def __init__(self, n_clique=3, n_state_clique=5, start=0):
        """
        Ring of cliques
        NOTES: No goals/rewards/etc since experiment was purely passive/unsupervised rep learning
        https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.community.ring_of_cliques.html#networkx.generators.community.ring_of_cliques
        """

        self.start = start
        self._access_matrix(n_clique, n_state_clique)
        super(RingOfCliques, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "clique-ring-graph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/clique-ring.png"

    def _access_matrix(self, n_clique=3, n_state_clique=5, p=0.0):
        """
        Sets the adjacency/stochastic matrix for the ring of cliques graph.
        INPUTS:  n_clique = number of cliques
                 n_state_clique = number of states per clique
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        """
        from networkx.generators.community import ring_of_cliques

        self.n_clique = n_clique
        self.n_state_clique = n_state_clique
        self.n_state = n_clique * n_state_clique
        self.G = ring_of_cliques(n_clique, n_state_clique).to_undirected()
        self.A = nx.adjacency_matrix(self.G).toarray()
        self.A_adj = self.A
        self.T = row_norm(self.A)

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions.
        NOTES: https://networkx.github.io/documentation/networkx-1.9/reference/drawing.html
        """
        self.layout = nx.shell_layout(self.G)
        self.info_state.loc[self.layout.keys(), "x"] = [
            self.layout[i][0] for i in self.layout.keys()
        ]
        self.info_state.loc[self.layout.keys(), "y"] = [
            self.layout[i][1] for i in self.layout.keys()
        ]
        self.xy = np.zeros((self.n_state, 2))
        self.xy[:, 0] = self.info_state.loc[:, "x"]
        self.xy[:, 1] = self.info_state.loc[:, "y"]
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    def draw_graph(
        self,
        node_size=8,
        edge_width=0.2,
        alpha=0.3,
        with_labels=False,
        color_communities=False,
        ax=None,
    ):
        """uses networkx to draw graph"""
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca()

        pos = self._pos_dict()
        if with_labels:
            nx.draw(
                self.G.to_undirected(),
                pos,
                node_size=node_size,
                linewidths=edge_width,
                edgecolors="black",
                node_color="black",
                font_size=24,
                ax=ax,
                with_labels=with_labels,
            )
        else:
            nx.draw_networkx_nodes(
                self.G.to_undirected(),
                pos,
                node_size=node_size,
                linewidths=0,
                edgecolors="black",
                node_color="black",
                ax=ax,
            )
            nx.draw_networkx_edges(
                self.G.to_undirected(),
                pos,
                edge_color="k",
                width=edge_width,
                alpha=alpha,
            )
        # plt.tight_layout()
        ax.axis("equal")
        ax.axis("off")


class AdjacencyEnv(GraphEnv):
    def __init__(self, A, xy):
        """ A = adjacency matrix, xy = vertex positions """
        self.A = A
        self.xy = xy
        self.start = 0
        self.n_state = self.A.shape[0]
        super(AdjacencyEnv, self).__init__()
        self._state_information()
        self.__name__ = "adjacency-graph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/adjacency_graph.png"

    def plot_state_func(
        self,
        state_vals,
        vlims=[None, None],
        ax=None,
        annotate=False,
        cmap=plt.cm.autumn,
        cbar=False,
        cbar_label="",
        node_edge_color="black",
        **kwargs
    ):
        """
        FUNCTION: plots state function state_vals on world_array imshow.
        INPUTS: state_vals = state values of function to plot
                vlims = [vmin,vmax] value range
                ax = figure axis to plot to
                annotate = textualize function values on states
                cmap = colormap
                cbar = include offset colorbar
                cbar_label = label for colorbar
        NOTES: uses mplot3d triangulated surface plots
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.axes(ax)
        triang = tri.Triangulation(self.xy[:, 0], self.xy[:, 1])
        plt.tricontourf(triang, state_vals, cmap=cmap, levels=100)

        if annotate:
            for state in range(self.n_state):
                x = self.xy[state, 0]
                y = self.xy[state, 1]
                state_val = state_vals[state]
                if not np.isnan(state_val):
                    text = ax.text(x, y, state_val, ha="center", va="center", color="k")
        remove_axes(ax)
        ax.axis("equal")
        if cbar:
            fig = plt.gcf()
            cbar = fig.colorbar(nc, shrink=0.6, orientation="horizontal", pad=0)
            if cbar_label != "":
                cbar.set_label(cbar_label)
        # plt.tight_layout()
        return ax


class SchapiroCommunityGraph(GraphEnv):
    def __init__(self, start=0):
        """
        Community graph environment (as in Schapiro2013)
        NOTES: No goals/rewards/etc since experiment was purely passive/unsupervised rep learning
        """
        self.n_state = 15
        self.start = start
        self._access_matrix()
        super(SchapiroCommunityGraph, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "community-graph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/community_graph.png"

    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        NOTES: Wrt Fig. 1, Schapiro2013, nodes are numbered 0-14 clockwise starting from leftmost entrance node to purple community.
        """
        A = np.zeros((self.n_state, self.n_state))
        # communities
        comm0 = range(5)
        comm1 = range(5, 10)
        comm2 = range(10, 15)
        comms = [comm0, comm1, comm2]
        # bnecks
        bnecks0 = [0, 4]
        bnecks1 = [5, 9]
        bnecks2 = [10, 14]
        bnecks = [bnecks0, bnecks1, bnecks2]
        # edges
        for comm in comms:
            for i in comm:
                for j in comm:
                    if i != j:
                        A[i, j] = 1.0
        # break within-community bottlenecks
        for i in range(3):
            A[bnecks[i][0], bnecks[i][1]] = 0.0
            A[bnecks[i][1], bnecks[i][0]] = 0.0
        # cross-community bottlenecks
        A[bnecks[0][1], bnecks[1][0]] = 1.0
        A[bnecks[1][1], bnecks[2][0]] = 1.0
        A[bnecks[2][1], bnecks[0][0]] = 1.0
        self.comms = comms
        self.bnecks = bnecks
        self.A = ((A + A.T) != 0).astype("int")

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions and communities/bnecks.
        """
        xy = np.zeros((self.n_state, 2))
        # define positions for single community then rotate/translate for others
        # defining positions for top middle community
        # http://mathworld.wolfram.com/RegularPentagon.html
        c1 = 0.25 * (np.sqrt(5) - 1)
        c2 = 0.25 * (np.sqrt(5) + 1)
        s1 = 0.25 * np.sqrt(10 + 2 * np.sqrt(5))
        s2 = 0.25 * np.sqrt(10 - 2 * np.sqrt(5))
        xy[0, :] = [-s2, -c2]
        xy[1, :] = [-s1, c1]
        xy[2, :] = [0, 1]
        xy[3, :] = [s1, c1]
        xy[4, :] = [s2, -c2]
        xy[0:5, :] = xy[0:5, :] - xy[0, :]  # translate so 0 coordinate at (0,0)
        # rotating positions for top middle community 135 degrees clockwise
        xy[5:10, :] = rotate_around_point(
            xy[0:5, :], radians=(135 / 180.0) * np.pi, origin=(0, 0)
        )
        # translating down
        xy[5:10, :] = xy[5:10, :] + np.array([2, -1 + 0.2])

        # rotating positions for top middle community 135 degrees anticlockwise
        xy[10:15, :] = rotate_around_point(
            xy[0:5, :], radians=-(135 / 180.0) * np.pi, origin=(0, 0)
        )
        # translating down
        xy[10:15, :] = xy[10:15, :] + np.array([0, -1.83 + 0.2])

        self.xy = xy
        self.info_state.loc[:, "x"] = xy[:, 0]
        self.info_state.loc[:, "y"] = xy[:, 1]
        self.info_state.loc[0:5, "community"] = 0
        self.info_state.loc[5:10, "community"] = 1
        self.info_state.loc[10:15, "community"] = 2
        self.info_state["bottleneck"] = False
        self.info_state["color"] = "black"
        self.info_state.loc[[0, 4, 5, 9, 10, 14], "bottleneck"] = True
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self, X=None):
        """Defines networkx graph including info_state information"""
        if X is None:
            # extract node/edge attributes
            nodesdf = self.info_state.reset_index()
            edgesdf = self.info_transition
            G = nx.from_pandas_edgelist(
                df=edgesdf, source="source", target="target", edge_attr="weight"
            )
            nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
            nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
            nx.set_node_attributes(
                G, name="community", values=nodesdf.community.to_dict()
            )
            nx.set_node_attributes(
                G, name="bottleneck", values=nodesdf.bottleneck.to_dict()
            )
            self.G = G
        else:
            self.G = nx.DiGraph(X)
            self._check_statespace()

    def draw_graph(
        self,
        node_size=100,
        linewidths=1,
        with_labels=False,
        color_communities=False,
        ax=None,
    ):
        """uses networkx to draw graph"""
        if ax is None:
            plt.figure(figsize=(5, 5))
            ax = plt.gca()

        pos = self._pos_dict()
        if color_communities:
            nodes_comm0 = [
                n
                for (n, ty) in nx.get_node_attributes(self.G, "community").items()
                if ty == 0
            ]
            nodes_comm1 = [
                n
                for (n, ty) in nx.get_node_attributes(self.G, "community").items()
                if ty == 1
            ]
            nodes_comm2 = [
                n
                for (n, ty) in nx.get_node_attributes(self.G, "community").items()
                if ty == 2
            ]
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=nodes_comm0,
                node_size=node_size,
                linewidths=linewidths,
                edgecolors="black",
                node_color="purple",
                with_labels=with_labels,
                font_size=24,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=nodes_comm1,
                node_size=node_size,
                linewidths=linewidths,
                edgecolors="black",
                node_color="green",
                with_labels=with_labels,
                font_size=24,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=nodes_comm2,
                node_size=node_size,
                linewidths=linewidths,
                edgecolors="black",
                node_color="darkorange",
                with_labels=with_labels,
                font_size=24,
                ax=ax,
            )
            nx.draw_networkx_edges(self.G, pos, edge_color="k", width=2, alpha=0.2)
        else:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                node_size=node_size,
                linewidths=linewidths,
                edgecolors="black",
                node_color="black",
                with_labels=with_labels,
                font_size=24,
                ax=ax,
            )
            nx.draw_networkx_edges(self.G, pos, edge_color="k", width=2, alpha=0.5)
        plt.tight_layout()
        ax.axis("equal")
        ax.axis("off")


class OBH2C(GraphEnv):
    def __init__(self, start=0):
        """
        Fig 2c state-space in Optimal Behavioral Hierarchy paper
        NOTES: No starts/goals/rewards/etc since experiment was purely passive/unsupervised rep learning
        """
        self.n_state = 10
        self.start = start
        self._access_matrix()
        super(OBH2C, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "OBH2C"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/OBH2C_graph.png"

    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        NOTES: Wrt Fig. 2c, Solway2015, nodes are numbered 0-9 clockwise starting from top->bottom, left->right.
        """
        A = np.zeros((self.n_state, self.n_state))
        A[0, 1] = 1
        A[0, 2] = 1
        A[0, 3] = 1
        A[1, 2] = 1
        A[2, 3] = 1
        A[3, 4] = 1
        A[4, 1] = 1
        A[4, 5] = 1
        A[5, 6] = 1
        A[6, 7] = 1
        A[7, 8] = 1
        A[8, 5] = 1
        A[6, 9] = 1
        A[7, 9] = 1
        A[8, 9] = 1

        # communities
        comm0 = range(5)
        comm1 = range(5, 10)

        self.comms = [comm0, comm1]
        self.bnecks = [4, 5]
        self.A = ((A + A.T) != 0).astype("int")

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions and communities/bnecks.
        """
        xy = np.zeros((self.n_state, 2))
        xy[0, :] = [0, 0]
        xy[1, :] = [1, 0]
        xy[2, :] = [0.5, -0.5]
        xy[3, :] = [0, -1]
        xy[4, :] = [1, -1]
        xy[5, :] = [1.5, -1.5]
        xy[6, :] = [2.5, -1.5]
        xy[7, :] = [2.0, -2.0]
        xy[8, :] = [1.5, -2.5]
        xy[9, :] = [2.5, -2.5]

        self.xy = xy
        self.info_state.loc[:, "x"] = xy[:, 0]
        self.info_state.loc[:, "y"] = xy[:, 1]
        self.info_state.loc[self.comms[0], "community"] = 0
        self.info_state.loc[self.comms[1], "community"] = 0
        self.info_state["bottleneck"] = False
        self.info_state.loc[self.bnecks, "bottleneck"] = True
        self.info_state["color"] = "black"
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self):
        """Defines networkx graph including info_state information"""
        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition
        G = nx.from_pandas_edgelist(
            df=edgesdf, source="source", target="target", edge_attr="weight"
        )
        nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
        nx.set_node_attributes(G, name="community", values=nodesdf.community.to_dict())
        nx.set_node_attributes(
            G, name="bottleneck", values=nodesdf.bottleneck.to_dict()
        )
        self.G = G


class GarvertGraph(GraphEnv):
    def __init__(self, start=0):
        """
        Fig 1 state-space in Garvert/Dolan/Behrens (2017) paper
        NOTES: No starts/goals/rewards/etc since experiment was purely passive/unsupervised rep learning
        """
        self.n_state = 12
        self.start = start
        self._access_matrix()
        super(GarvertGraph, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "GarvertGraph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/garvert2017_graph.png"

    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        NOTES: Wrt Fig. 1, Garvert2017, nodes are numbered 0-11 clockwise starting from top->bottom, left->right.
        """
        A = np.zeros((self.n_state, self.n_state))
        A[0, [1, 2, 3]] = 1
        A[1, [3, 4]] = 1
        A[2, [3, 5, 6]] = 1
        A[3, [4, 6, 7]] = 1
        A[4, [7, 8]] = 1
        A[5, [6, 9]] = 1
        A[6, [9, 10]] = 1
        A[7, [8, 10, 11]] = 1
        A[8, [11]] = 1
        A[9, [10]] = 1
        A[10, [11]] = 1
        self.A = ((A + A.T) != 0).astype("int")  # symmetrize

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions and communities/bnecks.
        """
        xy = np.zeros((self.n_state, 2))
        xy[0, :] = [-0.5, 0]
        xy[1, :] = [0.5, 0]

        xy[2, :] = [-1, -1]
        xy[3, :] = [0, -1]
        xy[4, :] = [1, -1]

        xy[5, :] = [-1.5, -2]
        xy[6, :] = [-0.5, -2]
        xy[7, :] = [0.5, -2]
        xy[8, :] = [1.5, -2]

        xy[9, :] = [-1, -3]
        xy[10, :] = [0, -3]
        xy[11, :] = [1, -3]

        self.xy = xy
        self.info_state.loc[:, "x"] = xy[:, 0]
        self.info_state.loc[:, "y"] = xy[:, 1]
        self.info_state["color"] = "black"
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self):
        """Defines networkx graph including info_state information"""
        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition
        G = nx.from_pandas_edgelist(
            df=edgesdf, source="source", target="target", edge_attr="weight"
        )
        nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
        self.G = G


class OpenBox(GridWorld):
    """
    Openbox grid world.
    INPUTS: scale = length of side of box
            start/goal/name/reward/stepcost/goal_absorb/stay_actions as in other envs
    """

    def __init__(
        self,
        scale=10,
        start=None,
        goal=None,
        name="open-box",
        reward=10,
        stepcost=1,
        goal_absorb=False,
        stay_actions=False,
        **kwargs
    ):
        self.__name__ = name
        self.scale = scale
        self.start_center = int(
            np.ceil(self.scale / 2.0 - 1) * self.scale + np.ceil(self.scale / 2.0 - 1)
        )
        self.start_default = self.start_center
        if start is None:
            self.start = self.start_default
        else:
            self.start = start
        super(OpenBox, self).__init__(
            world_array=self.gridworld_openbox(scale=scale),
            start=self.start,
            goal=goal,
            name=name,
            reward=reward,
            stepcost=stepcost,
            goal_absorb=goal_absorb,
            stay_actions=stay_actions,
            **kwargs
        )

    def gridworld_openbox(self, scale=10, diag_move=True):
        """scale is the number of open states along each wall"""
        world_array = np.zeros((scale + 2, scale + 2))
        world_array[0, :] = 1.0
        world_array[:, 0] = 1.0
        world_array[-1, :] = 1.0
        world_array[:, -1] = 1.0
        return world_array


class TJunction(GridWorld):
    """
    T-junction grid world.
    INPUTS: start/goal/name/reward/stepcost/goal_absorb/stay_actions as in other envs
    """

    def __init__(self, complete_circuit=False, directed=False, backflow=0.05, **kwargs):
        super(TJunction, self).__init__(
            world_array=self.gridworld_junctionmaze(complete_circuit=complete_circuit),
            **kwargs
        )
        self.__name__ = "t-junction"
        self.directed = directed
        self.complete_circuit = complete_circuit
        self.backflow = backflow
        # viz properties
        self.env_lw = 1
        if self.directed:
            A = np.zeros((self.n_state, self.n_state))
            A[0, 7] = 1
            A[7, 10] = 1
            A[10, 13] = 1
            A[13, 16] = 1
            A[16, 19] = 1
            A[19, 22] = 1
            A[22, 25] = 1
            A[26, 23] = 1
            A[23, 20] = 1
            A[20, 17] = 1
            A[17, 14] = 1
            A[14, 11] = 1
            A[11, 8] = 1
            A[8, 3] = 1
            A[3, 2] = 1
            A[2, 1] = 1
            A[1, 0] = 1
            A[3, 4] = 1
            A[4, 5] = 1
            A[5, 6] = 1
            A[6, 9] = 1
            A[9, 12] = 1
            A[12, 15] = 1
            A[15, 18] = 1
            A[18, 21] = 1
            A[21, 24] = 1
            A[24, 27] = 1
            if self.complete_circuit:
                A[25, 25] = 0  # unabsorb
                A[25, 28] = 1
                A[28, 29] = 1
                A[29, 30] = 1
                A[30, 31] = 1
                A[31, 26] = 1
                A[27, 27] = 0  # unabsorb
                A[27, 34] = 1
                A[34, 33] = 1
                A[33, 32] = 1
                A[32, 31] = 1
                A[31, 26] = 1
            # else:
            #     # back to start
            #     A[25,26] = 1
            #     A[27,26] = 1
            self.A = A
            self.T = deepcopy(A.astype("float"))
            if self.backflow > 0:
                self.A += self.A.T
                self.T += self.backflow * self.T.T
            if not self.complete_circuit:
                # add absorbing goals
                self.A[25, 25] = 1
                self.A[27, 27] = 1
                self.T[25, 25] = 1
                self.T[27, 27] = 1
            self.T = row_norm(self.T)

    def gridworld_junctionmaze(self, complete_circuit=False):
        """T-maze junction type state-space"""
        if complete_circuit:
            world_array = np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            )
        else:
            world_array = np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            )
        return world_array


class LinearTrack(GridWorld):
    def __init__(self, scale=10, goal=None, goal_absorb=False, name="linear-track"):
        """
            Linear track environment
            INPUTS: scale = number of states in track
                    goal = whether or not a goal is present and transition structure encodes optimal policy
                    inputs as in GridWorld
        """
        self.n_state = scale
        self.reward = 10
        self.stepcost = 1
        self.start = 0
        self.start_center = int(np.floor(scale / 2.0))
        self.start_default = self.start
        self.goal = goal
        self.goal_absorb = goal_absorb
        self._access_matrix(goal=self.goal)
        self.__name__ = name
        super(LinearTrack, self).__init__(
            world_array=self.gridworld_lineartrack(scale=scale),
            start=self.start,
            goal=self.goal,
            name=self.__name__,
            reward=self.reward,
            stepcost=self.stepcost,
            goal_absorb=self.goal_absorb,
        )
        self._state_information()
        self._node_info()
        self._set_graph()
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.n_dim = 2
        self.fname_graph = "figures/linear_track.png"

    def gridworld_lineartrack(self, scale=10):
        """scale is the number of open states along each wall"""
        world_array = np.zeros((3, scale + 2))
        world_array[0, :] = 1.0
        world_array[:, 0] = 1.0
        world_array[-1, :] = 1.0
        world_array[:, -1] = 1.0
        return world_array

    def _access_matrix(self, goal=None):
        """Set the adjacency/stochastic matrix for a linear track.

        Args:
          goal = state number or None
        Sets:
          A/T: adjacency/stochastic matrix
        """
        if goal is not None:
            assert goal < self.n_state, "Mis-specified goal state"
            self.n_goal_state = 1
        else:
            self.n_goal_state = 0
        self.n_state_via = self.n_state - self.n_goal_state

        A = np.zeros((self.n_state, self.n_state))
        for i in range(0, self.n_state - 1):
            A[i, i + 1] = 1.0
        # adjacency matrix of underlying graph
        self.A = A + A.T
        self.A_adj = self.A.copy()
        # stochastic matrix
        if goal is not None:
            # optimal policy
            self.T = np.zeros((self.n_state, self.n_state))
            for i in range(self.n_state):
                if i < self.goal:
                    self.T[i, i + 1] = 1.0
                elif i == self.goal:
                    if i > 0:
                        self.T[i, i - 1] = 0.5
                    if i < self.n_state - 1:
                        self.T[i, i + 1] = 0.5
                else:
                    if i > 0:
                        self.T[i, i - 1] = 1.0
            self.T = row_norm(self.T)

    def _set_graph(self):
        """Defines networkx graph including info_state information"""
        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition
        G = nx.from_pandas_edgelist(
            df=edgesdf, source="source", target="target", edge_attr="weight"
        )
        nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
        self.G = G

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions and communities/bnecks.
        """
        xy = np.zeros((self.n_state, 2))
        for i in range(self.n_state):
            xy[i] = np.array([i + 1, 1])
        self.info_state.loc[:, "x"] = xy[:, 0]
        self.info_state.loc[:, "y"] = xy[:, 1]
        self.info_state["color"] = "black"
        self.xy = xy
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    def draw_graph(self):
        """uses networkx to draw graph"""
        ax = self.plot_environment(figsize=(10, 10), fname=None)

        pos = self._pos_dict()
        nx.draw(
            self.G,
            pos,
            node_size=30,
            alpha=0.3,
            node_color="blue",
            with_labels=True,
            font_size=24,
            ax=ax,
        )
        plt.axis("equal")
