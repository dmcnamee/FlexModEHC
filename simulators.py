#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np; np.seterr(all='raise')
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm
from copy import deepcopy

import config
from visualization import add_jitter, plot_state_func, figsize, cmap_state_density
from utils import norm_density, ensure_dir
from timer import timeit_debug, timeit_info
from numba import jit, prange
from autocorrelation import estimate_occ_zero_cf, estimate_occ_acf


random_state = np.random.RandomState(1234)


class Simulator(object):
    """
    FUNCTION: Samples from propagator.
    INPUTS: rho_init        = initial state density, None = random selection, 'env_default' = ENV.start
            mass            = assume particle as momentum, approximately corresponds to #previous states to avoid
                            # TODO reconceptualize as self-motion cue
            no_dwell        = True, sample away from current state (i.e. force jump, ->embedded discrete-time chain)
            diagnostics     = computes coverage efficiency/MSD etc of sampled trajectories
    """
    def __init__(self, PROP, rho_init='env_default', mass=1, no_dwell=True, label='SIMULATOR', target_dir='simulations', **kwargs):
        self.PROP = PROP
        if rho_init == 'env_default':
            self.rho_init = process_rho(self.ENV.start, self.n_state)
        else:
            self.rho_init = process_rho(rho_init, self.n_state)
        self.no_dwell = no_dwell
        self.mass = mass
        self.label = label
        self.target_dir = target_dir
        self.sequences_sampled = False
        self.set_viz_scheme()
        if hasattr(self.ENV, 'R'):
            self.no_reward_func = False # sampling functions will sample rewards
        else:
            self.no_reward_func = True # sampling functions will not sample rewards
        for key, value in kwargs.items():
            setattr(self, key, value)

        np.random.seed()
        assert self.rho_init.size == self.PROP.n_state, 'Dimensionality of starting state density is incorrect.'
        self._set_file_names()

    @timeit_debug
    def _set_file_names(self):
        self.output_dir = ensure_dir(os.path.join(self.target_dir, self.ENV.__name__))
        self.fname_base = self.ENV.__name__
        self.fname_vector = os.path.join(self.output_dir, 'density_' + self.fname_base + self.label +'.csv')
        self.fname_scalar = os.path.join(self.output_dir, 'var_' + self.fname_base + self.label +'.csv')
        self.fname_state = os.path.join(self.output_dir, 'state_' + self.fname_base + self.label +'.csv')
        self.fname_learn = os.path.join(self.output_dir, 'learn_' + self.fname_base + self.label +'.csv')
        self.fname_explore = os.path.join(self.output_dir, 'explore_' + self.fname_base + self.label +'.csv')
        self.fname_class = os.path.join(self.output_dir, self.fname_base + self.label + '.sim')

    @property
    def beta(self):
        return self.PROP.beta

    @property
    def n_dim(self):
        return self.PROP.n_dim

    @property
    def n_state(self):
        return self.PROP.n_state

    @property
    def ix_states(self):
        return self.PROP.states

    @property
    def world_array(self):
        return self.GEN.ENV.world_array

    @property
    def GEN(self):
        return self.PROP.GEN

    @property
    def ENV(self):
        return self.PROP.GEN.ENV

    @ENV.setter
    def ENV(self, value):
        self.PROP.GEN.ENV = value

    @timeit_debug
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def evolve(self, n_step=1, rho_start=None, ignore_imag=True):
        """
        Evolves prior state density rho_start forward to rho_stop using self.PROP
        INPUTS: rho_start = prior state density to evolve
                n_step = number of steps to evolve
                ignore_imag = False, raises error
        NOTES: N/A
        """
        if rho_start is None:
            rho_start = self.rho_init
        self._check_state_density(rho_state=rho_start)
        for i in range(n_step):
            rho_stop = np.dot(rho_start, self.PROP.etO)
            if not np.all(np.isreal(rho_stop)):
                if ignore_imag:
                    print('SIMULATOR: complex propagated density')
                    rho_stop = rho_stop.real
                else:
                    raise ValueError('SIMULATOR: complex propagation density')
            rho_stop = norm_density(rho_stop, beta=self.beta, type='l1') # L1 normalize
            rho_start = rho_stop
        self._check_state_density(rho_state=rho_start)
        return rho_start

    @timeit_debug
    def _check_state_density(self, rho_state, n_decimals=config.n_decimals):
        """Checks whether rho_state is a probability density (within a precision threshold)."""
        assert (rho_state>=0).all(), 'State density in negative range.'
        if np.allclose(rho_state.sum(),1):
            return rho_state/rho_state.sum()
        else:
            raise ValueError('SIMULATOR: state density sums to %.8f (!= 1).'%rho_state.sum())

    @timeit_debug
    def _sample_state(self, rho_state, prev_states=None):
        """
        FUNCTION: Samples from propagated state density.
        INPUTS: rho_state = propagated state density
                prev_states = recently occupied states
        NOTES: Use no_dwell and prev_states to endow sampling with "momentum" away from current position
        """
        if (self.no_dwell and self.mass != 0.) and prev_states is not None:
            # endows sampling with momentum akin to a self.motion cue during path integration
            mass = np.floor(self.mass).astype('int')
            states = prev_states
            except_states = states[~np.isnan(states)].astype('int')
            except_states = except_states[-np.min([mass,len(except_states)]):] # "memory" of states
            rho_state[except_states] = 0.
        rho_state = rho_state/rho_state.sum()
        return sample_discrete(rho_state)

    @timeit_debug
    def _sample_reward(self, state, prev_state=None):
        """
        FUNCTION: Samples reward at state based on ENV.R.
        INPUTS: state       = state at which reward observed
                prev_state  = previous state
        NOTES: N/A
        """
        if prev_state is None:
            reward = self.ENV.R_state[state]
        else:
            reward = self.ENV.R[prev_state,state]
        return reward

    @timeit_info
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def sample_sequences(self, n_step=100, n_samp=50, rho_start=None, fast_storage=True):
        """
        FUNCTION: Returns sequences of sampled states defining state-space trajectories.
        INPUTS:
            n_step        = number of timesteps in each trajectory sample
            n_samp        = number of sequence samples
            rho_start     = state or state density to start from (defaults to self.rho_init)
            fast_storage  = True: store as numpy arrays, False: pandas dataframes
        """
        self.sequences_generated = True
        self.fast_storage = fast_storage
        if rho_start is None:
            rho_start = self.rho_init
        rho_start = process_rho(rho_start, self.n_state)
        n_seq_steps = n_step + 1 # include initial state
        self.n_step = n_step
        self.n_samp = n_samp
        self.n_seq_steps = n_seq_steps
        self.ix_steps = np.arange(0,n_seq_steps,1)
        self.ix_samps = np.arange(0,n_samp,1)
        self.samp_times = self.ix_steps*(1/self.PROP.tau)
        self._set_output_container()
        self.ix_slice = pd.IndexSlice

        # loop over trajectories
        if config.verbose:
            iterator = tqdm(range(n_samp), desc='SAMPLING')
        else:
            # iterator = prange(n_samp)
            iterator = range(n_samp)
        state_seqs = np.zeros((n_samp, n_seq_steps))
        rhos = np.zeros((n_samp, n_seq_steps, self.n_state))
        rewards = np.zeros((n_samp, n_seq_steps))
        for ns in iterator:
            state = self._sample_state(rho_start)
            state_seqs[ns,0] = state
            rhos[ns,0,:] = rho_start # note rho_stop convention at step 0
            if not self.no_reward_func:
                rewards[ns,0] = self._sample_reward(state)
            # loop within trajectories
            rho_inter = process_rho(state, self.n_state) # sampled state as prior for next step
            for n in range(1,n_seq_steps):
                rho_stop = self.evolve(n_step=1, rho_start=rho_inter)
                state = self._sample_state(rho_stop, prev_states=np.array(state_seqs[ns,:n]))
                state_seqs[ns,n] = state
                rhos[ns,n,:] = rho_stop
                if not self.no_reward_func:
                    rewards[ns,n] = self._sample_reward(state)
                # sampled state as prior for next step
                rho_inter = process_rho(state, self.n_state)
        if fast_storage:
            self.state_seqs = state_seqs.astype('int')
            self.rhos = rhos
            self.rewards = rewards
        else:
            self.output_scalar.loc[self.ix_slice[:,:],'state'] = state_seqs.flatten()
            self.output_vector.loc[self.ix_slice[:,:,:],'rho_stop'] = rhos.flatten()
            self.output_scalar.loc[self.ix_slice[:,:],'reward'] = rewards.flatten()


    @timeit_info
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def sequences_to_paths(self):
        """
        FUNCTION: "fill ins" jumped states and removes auto-sampled dates to derive complete paths from sampled sequences
        OUTPUTS: self.paths = list of "filled-in" and "smoothed" paths
                 self.paths_array = array of paths of minimal length (across sampled sequences)
        """
        # n_samp x n_seq_steps matrix of sequence samples
        state_seqs = self._retrieve_state(samp=None, step=None, coords=False)
        self.paths = []
        min_length = np.inf
        for samp in range(self.n_samp):
            ix1 = 0; ix2 = ix1 + 1
            state_start = state_seqs[samp,ix1]
            path = [state_start]
            while (ix1 < self.n_seq_steps) and (ix2 < self.n_seq_steps):
                state_start = state_seqs[samp,ix1]
                state_end = state_seqs[samp,ix2]
                while state_start == state_end:
                    ix2 += 1
                    if ix2 < self.n_seq_steps:
                        state_end = state_seqs[samp,ix2]
                    else:
                        break
                path_comp = nx.shortest_path(self.ENV.G, source=state_start, target=state_end)
                path += path_comp[1:] # remove start state_start
                ix1 = ix2
                ix2 = ix1 + 1
            self.paths.append(path)
            min_length = np.min([min_length, len(path)]).astype('int')
        # assert min_length-1>1, 'interpolated path lengths = 1, agent does not move from initial state?'
        print('SIMULATOR.sequences_to_paths: paths truncated to minimum length of %i.'%(min_length-1))
        self.paths_array = np.zeros((self.n_samp, min_length))
        for samp in range(self.n_samp):
            self.paths_array[samp,:] = self.paths[samp][:min_length]
        self.path_length = min_length

    @timeit_info
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def path_efficiency(self, step_cost=1.):
        """
        FUNCTION: counts number of unique states visited as a function of path length and compares vs cumulative step cost
        INPUT: step_cost = cost (e.g. time/distance) per step
               self.paths_array = array of "filled-in" and smoothed sequences
        OUTPUT: self.path_count = number of unique states at each step
                self.path_eff = number of unique states divided by cumulative cost
        NOTES: parallelized over sampled sequences
        """
        if not hasattr(self, 'paths_array'):
            self.sequences_to_paths()
        self.path_count = np.zeros((self.path_length,))
        self.path_eff = np.zeros((self.path_length,))
        for step in range(self.path_length-1):
            vis_states = np.unique(self.paths_array[:,:step+1]).tolist()
            self.path_count[step] = np.isin(self.ix_states, vis_states).sum() # check distinct states
        self.path_step_cost = step_cost
        self.path_cost = np.arange(self.path_length)*step_cost*self.n_samp
        self.path_eff[0] = 0. # initial path efficiency set to 0 by default
        self.path_eff[1:] = self.path_count[1:]/self.path_cost[1:]

    @timeit_info
    def sample_limit_density(self, rho_start, n_step=10, n_samp=10):
        """
        FUNCTION: Sample from limiting density.
        INPUTS: rho_start   = start state distribution (defaults to rho_init)
                n_step      = number of steps
                n_samp      = number of sequences
        NOTES: Samples from top eigenvector self.kernels[:,0]
        """
        rho_limit = norm_density(self.kernels[:,0], beta=self.beta)
        if rho_start == None:
            rho_start = self.rho_init
        rho_start = process_rho(rho_start, self.n_state)
        # loop over trajectories
        for ns in prange(n_samp):
            state = self._sample_state(rho_start)
            self._record_state(samp=ns, step=0, state=state)
            self._record_vector(samp=ns, step=0, rho_stop=rho_start) # note rho_stop convention at step 0
            # loop within trajectories
            for n in range(1,self.n_seq_steps):
                state = self._sample_state(rho_limit)
                self._record_vector(samp=ns, step=n, rho_stop=rho_limit)
                self._record_state(samp=ns, step=n, state=state)

    @timeit_debug
    def _set_output_container(self):
        """
        FUNCTION: Establishes pandas dataframes for storing results.
        NOTES: output_scalar records scalar variables for every trajectory sample/step
               output_vector records state vector variables for every trajectory sample/step
               output_sim records summary diagnostics/statistics per step over trajectories
        """
        vectorIX = pd.MultiIndex.from_product([self.ix_samps, self.ix_steps, self.ix_states], names=['sample', 'step', 'state'])
        scalarIX = pd.MultiIndex.from_product([self.ix_samps, self.ix_steps], names=['sample', 'step'])
        stateIX = pd.Index(data=self.ix_states, name='state')

        # records and diagnostics containers
        self.output_vector = pd.DataFrame(data=np.nan, index=vectorIX, columns=['rho_start', 'rho_stop'], dtype='float')
        self.output_scalar = pd.DataFrame(data=np.nan, index=scalarIX, columns=['state']+['reward'], dtype='float')
        self.output_state = pd.DataFrame(data=self.ix_states, index=stateIX, columns=['id'], dtype='int')
        self.output_scalar.loc[:,'state'] = -1
        self.output_scalar = self.output_scalar.astype(dtype={'state':'int'})
        self.__copy_env_info()
        self.__copy_gen_info()
        self.__copy_prop_info()

    @timeit_debug
    def _record_scalar(self, samp, step, **kwargs):
        """Record scalar variables in self.output_scalar"""
        for key, value in kwargs.items():
            self.output_scalar.loc[self.ix_slice[samp,step],key] = value

    @timeit_debug
    def _record_vector(self, samp, step, **kwargs):
        """Record state vector variables in self.output_vector"""
        for key, value in kwargs.items():
            self.output_vector.loc[self.ix_slice[samp,step,:],key] = value

    def _record_state(self, samp, step, state):
         self._record_scalar(samp=samp, step=step, state=state)
         coords = self.ENV._retrieve_state_coordinates(state)
         self._record_scalar(samp=samp, step=step, x=coords[0], y=coords[1])

    def _record_reward(self, samp, step, reward):
         self._record_scalar(samp=samp, step=step, reward=reward)

    def _retrieve_vector(self, samp=slice(None), step=slice(None), state=slice(None), key='rho_stop'):
        """
        FUNCTION: Retrieve state vector in self.output_vector
        INPUT: samp = trajectory sample number
               step = step number, step=None returns entire trajectory
               state= state, state=None returns all states
               key  = column in self.output_vector to return
        """
        if samp == None:
            samp = slice(None)
        if step == None:
            step = slice(None)
        if state == None:
            state = slice(None)
        if self.fast_storage:
            if key == 'rho_stop':
                return self.rhos[samp, step, state]
            else:
                raise ValueError('key not available with fast_storage')
        else:
            self.output_vector.loc[self.ix_slice[samp,step,state],key]


    def _retrieve_scalar(self, samp, step=None, key='state'):
        """
        FUNCTION: Retrieve state scalar in self.output_vector
        INPUT: samp = trajectory sample number
               step = step number, step=None returns entire trajectory
               key  = column in self.output_vector to return
        """
        if step is None:
            return self.output_scalar.loc[self.ix_slice[samp,:],key]
        else:
            return self.output_scalar.loc[self.ix_slice[samp,step],key]

    def _retrieve_state(self, samp=None, step=None, coords=True):
        """
        FUNCTION: Retrieve state scalar in self.output_vector
        INPUT: samp = trajectory sample number
               step = step number, step=None returns entire trajectory
               coords = True, returns coords, otherwise returns state index
        """
        if self.fast_storage:
            if samp is None:
                if step is None:
                    state_seqs = self.state_seqs
                else:
                    state_seqs = self.state_seqs[:,step]
            else:
                if step is None:
                    state_seqs = self.state_seqs[samp,:]
                else:
                    state_seqs = self.state_seqs[samp,step]
        else:
            if samp is None:
                if step is None:
                    state_seqs = self.output_scalar.loc[self.ix_slice[:,:],'state'].unstack().values.reshape((self.n_samp,self.n_seq_steps)).astype('int') # samp x step shape
                else:
                    state_seqs = self.output_scalar.loc[self.ix_slice[:,step],'state'].unstack().values.reshape((self.n_samp,1)).astype('int') # samp x 1 shape
            else:
                if step is None:
                    state_seqs = self.output_scalar.loc[self.ix_slice[samp,:],'state'].unstack().values.reshape((1,self.n_seq_steps)).astype('int') # 1 x step shape
                else:
                    ix = self.ix_slice[samp,step]
                    state_seqs = self.output_scalar.loc[self.ix_slice[samp,step],'state'].unstack().values.astype('int') # 1 x 1 shape

        if coords:
            # convert states to coordinates
            if samp is None and step is None:
                # output shape is samp x step x coord
                state_coords = np.zeros((self.n_samp, self.n_seq_steps, self.n_dim))
                for samp in range(self.n_samp):
                    for step in range(self.n_seq_steps):
                        state_coords[samp,step,:] = self.ENV.xy[int(state_seqs[samp,step]),:]
            elif step is None:
                # output shape is step x coord
                state_coords = np.zeros((self.n_seq_steps, self.n_dim))
                for step in range(self.n_seq_steps):
                    state_coords[step,:] = self.ENV.xy[int(state_seqs[step]),:]
            else:
                raise ValueError('i have not prepared for this eventuality')
            return state_coords
        else:
            return state_seqs

    def stateseq_to_coords(self, state_seq):
        """converts state sequence to coordinates sequence"""
        n_seq_steps = len(state_seq)
        state_coords = np.zeros((n_seq_steps, self.n_dim))
        for step in range(n_seq_steps):
            state_coords[step,:] = self.ENV.xy[int(state_seq[step]),:]
        return state_coords

    def _retrieve_reward(self, samp, step=None):
        """
        FUNCTION: Retrieve reward observed at samp/step
        INPUT: samp = trajectory sample number
               step = step number, step=None returns entire trajectory
        """
        if self.fast_storage:
            if step is None:
                return self.rewards[samp,:]
            else:
                return self.rewards[samp,step]
        else:
            if step is None:
                return self.output_scalar.loc[self.ix_slice[samp,:],'reward']
            else:
                return self.output_scalar.loc[self.ix_slice[samp,step],'reward']

    def __copy_env_info(self):
        """Copies env/plot info and state type to self.output containers."""
        # TODO should be a one-liner for this, problem is index alignment
        for iter_n in self.ix_steps:
            if hasattr(self.ENV,'xy'):
                self.output_state.loc[:, 'x'] = self.ENV.xy[:, 0]
                self.output_state.loc[:, 'y'] = self.ENV.xy[:, 1]
            for var in self.ENV.info_state.columns:
                self.output_state.loc[:, var] = self.ENV.info_state[var].values

    def __copy_gen_info(self):
        """Copies generator info and state type to self.output containers."""
        pass

    def __copy_prop_info(self):
        """Copies propagator info and state type to self.output containers."""
        pass

    @timeit_info
    def _process_output(self, thresh=0.95):
        """Processes output e.g. computes diagnostics"""
        # fill in paths of sampled sequences
        self.state_paths = fill_in_paths(self.state_seq, self.GEN.W)
        if self.diagnostics:
            self.compute_diagnostics()
        print('todo')

    def set_viz_scheme(self, ax=None, **kwargs):
        """
        FUNCTION: Sets visualization setting.
        INPUTS: Can overwrite all settings (see below).
        """
        self.max_samps = 50
        self.max_steps = 100
        self.jitter_state = True # add jitter for plotting clarity
        self.jitter_std = 0.02
        self.state_msize = 2
        self.state_lw = 0.
        self.traj_width = 1.
        self.traj_format = '-o'
        self.traj_alpha = 1.
        self.color_time = True
        self.color_traj = 'gray'
        self.start_pos = False
        self.color_start = 'black'
        self.marker_start = 'x'
        self.cbar = False
        self.cmap_traj = plt.cm.YlOrRd # plt.cm.hot_r, plt.cm_autumn_r
        self.cmap_samp = 'pastel'
        self.cmap_state_density = plt.cm.bone
        self.cmap_grid_code = plt.cm.jet
        self.env_lw = 2 # environment linewidth
        self.file_ext = '.pdf'
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.set_target_axis(ax=ax)

    def set_target_axis(self, ax=None):
        """
        FUNCTION: Sets whether a fig/axis is to be used for plotting output.
        """
        if ax is None:
            self.fig = None
            self.ax = None
            self.no_target_axis = True
        else:
            self.fig = plt.gcf()
            self.ax = ax
            self.no_target_axis = False

    def fig_axis(self, hold=False, figsize=figsize):
        """
        FUNCTION: Sets figure and/or axis to plot to.
        """
        if self.no_target_axis:
            fig, ax = plt.subplots(figsize=figsize)
            self.fig = fig
            self.ax = ax
        if hold:
            # set as target axis, do not overwrite with subsequent fig_axis calls
            self.no_target_axis = False

    def plot_trajectory(self, state_seq=None, samp=0, plot_env=True, state_func_env=True, figsize=figsize):
        """Plots single trajectory on top of self.ENV.plot_environment (plot_env=True)."""
        self.fig_axis(figsize=figsize)
        if state_seq is None:
            coords = self._retrieve_state(samp=samp, step=None, coords=True)
        else:
            coords = self.stateseq_to_coords(state_seq=state_seq)
        self.max_steps = np.min([coords.shape[0], self.max_steps])
        coords = coords[:self.max_steps,:]
        if self.jitter_state:
            coords = add_jitter(coords, std=self.jitter_std)

        x = coords[:,0]
        y = coords[:,1]
        if plot_env:
            if state_func_env:
                self.ENV.plot_state_func(state_vals=np.zeros(self.n_state), ax=self.ax, annotate=False, interpolation='none', cmap=plt.cm.Greys, cbar=False, cbar_label='')
                if self.ENV.__type__ == 'roomworld' or self.ENV.__type__ == 'gridworld':
                    x = x-1
                    y = y-1
            else:
                self.ENV.plot_environment(ax=self.ax)
        elif state_func_env:
            x = x-1
            y = y-1

        # plot line connecting state samples
        self.ax.plot(x, y, self.traj_format,
                     color=self.color_traj,
                     markersize=self.state_msize,
                     linewidth=self.traj_width,
                     mew=self.state_lw,
                     alpha=self.traj_alpha,
                     zorder=101)

        if self.color_time:
            colors = self.cmap_traj(np.linspace(0, 1, self.max_steps))
            for xs, ys, c in zip(x, y, colors):
                self.ax.scatter(xs, ys, color=c, edgecolor='k', s=self.state_msize**2, linewidth=self.state_lw, zorder=102)

        if self.start_pos:
            start_coords = np.array([x[0], y[0]]).reshape((1,2))
            if self.jitter_state:
                start_coords = add_jitter(start_coords, std=self.jitter_std).flatten()
            else:
                start_coords = start_coords.flatten()
            self.ax.scatter(start_coords[0], start_coords[1], color=self.color_start, marker=self.marker_start, linewidth=4., s=self.state_msize**2, zorder=103)

    def plot_trajectories(self, samps=None, multi_colored=True):
        """
        FUNCTION: plots multiple trajectories
        INPUTS: samps = list of sequence samples to plot, =None plots all.
                multi_colored = color each sequence differently
        """
        self.fig_axis(hold=True)
        if samps is None:
            samps = range(self.n_samp)
        assert (np.array(samps) < self.n_samp).all(), 'Requested sequence samples out of range.'
        if multi_colored:
            cmap_samp = sb.color_palette(self.cmap_samp, len(samps))
        color_time_temp = self.color_time
        self.color_time = False
        color_traj_temp = self.color_traj
        for i,s in enumerate(samps):
            if multi_colored:
                self.color_traj = cmap_samp[i]
            self.plot_trajectory(samp=s)
        self.color_time = color_time_temp
        self.color_traj = color_traj_temp
        self.set_target_axis(ax=None)

    def plot_propagator_timeslice(self, step=1, samp=0, cbar=False):
        """
        FUNCTION: plots sampled state density
        """
        self.fig_axis()
        rho_stop = self._retrieve_vector(samp=samp, step=step, key='rho_stop')
        plot_state_func(rho_stop.values, self.world_array, cmap=cmap_state_density, cbar=cbar, ax=self.ax)
        self.ax.set_title('Sample %i, step %i'%(samp,step), y=1.)

    def plot_state_density(self, rho, cmap=cmap_state_density, **kwargs):
        """
        FUNCTION: plots given state density
        """
        self.fig_axis()
        self.ENV.plot_state_func(state_vals=rho, cmap=cmap, ax=self.ax, **kwargs)
        self.ax.set_title('state density', y=1.)

    @timeit_debug
    def save_output(self):
        """Saves output in "flat" format for plotting etc"""
        # flatten
        output_scalar = self.output_scalar.reset_index()
        output_vector = self.output_vector.reset_index()
        output_state = self.output_state.reset_index()
        # save
        output_scalar.to_csv(self.fname_scalar)
        output_vector.to_csv(self.fname_vector)
        output_state.to_csv(self.fname_state)

    @timeit_info
    def estimate_cf(self, dist_occ=0, zero_pos=False):
        """
        FUNCTION: estimates acf of sampled trajectories
        INPUTS: dist_occ = occupator distance, if two states are considered equivalent within a distance of dist_occ
                zero_pos = correlation with respect to zero-th position only
        """
        samps_state = self._retrieve_state(samp=None, step=None, coords=False)
        if zero_pos:
            self.acf_mean, self.acf_sem = estimate_occ_zero_cf(samps_state.T, d=dist_occ)
        else:
            self.acf_mean, self.acf_sem = estimate_occ_acf(samps_state.T, d=dist_occ)



    @timeit_debug
    def save(self, fname=None):
        """save Simulator"""
        # self.fig = None
        # self.ax = None
        # self.ENV = None
        self.save_output()
        file = open(self.fname_class,'wb')
        attrs = deepcopy(self.__dict__)
        del attrs['fig']
        del attrs['ax'] # cannot pickle figs/axes
        del attrs['PROP'].__dict__['GEN'].__dict__['ENV'] # cannot pickle maze environment
        file.write(pickle.dumps(attrs))
        file.close()

    @timeit_info
    def load(self):
        """load Simulator"""
        file = open(self.fname_class,'rb')
        dataPickle = file.read()
        file.close()
        dict_temp = deepcopy(self.__dict__)
        self.__dict__ = pickle.loads(dataPickle)


def process_rho(rho, n_state):
    """
    FUNCTION: Process state distribution.
    INPUTS: rho = state distribution or state
    OUTPUTS: rho_out = state distribution
    NOTES: rho = None returns a uniform distribution
           rho = state returns a one-hot distribution
           else rho_out==rho
    """
    if rho is None:
        rho_out = np.ones((n_state))
        return rho_out/rho_out.sum()
    elif not hasattr(rho, "__len__") or np.array(rho).size < n_state:
        rho_out = np.zeros((n_state))
        rho_out[np.asarray(rho).astype('int')] = 1.
    else:
        rho_out = rho
    return rho_out/rho_out.sum()

def sample_discrete(p):
    """FUNCTION: discrete sample from 1:len(p) with prob p."""
    return np.random.choice(list(range(len(p))), 1, p=p)


def fill_in_paths(state_seq, W):
    """
    FUNCTION: ``fills in'' state sequences using shortest paths.
    INPUT: state_seq = nsamples x seq length matrix of state sequences
           W         = weight/adjacency matrix
    OUTPUT: state_paths = list (length nsamples) of "completed" paths
    NOTES: only effects non-random walks
    """
    n_samp = state_seq.shape[0]
    state_paths = []
    for ns in range(n_samp):
        state_paths.append(fill_in_path(state_seq[ns,:], W))
    return state_paths


def fill_in_path(state_seq, W):
    """
    FUNCTION: ``fills in'' state sequence using shortest paths.
    INPUT: state_seq = state sequence (with possibly non-adjacent jumps)
           W = weight/adjacency matrix
    OUTPUT: path = "filled in" state sequence
    """
    n_seq_steps = len(state_seq)
    path = deepcopy(state_seq)
    step = 0
    for n in range(n_seq_steps-1):
        if W[state_seq[n],state_seq[n+1]]==0:
            # get shortest path
            subpath = shortest_path(W, state_seq[n], state_seq[n+1])
            # insert into sequence
            subpath = subpath[1:-1]
            path = np.insert(path,step+1,subpath)
            step = step + len(subpath)
        step = step + 1
    return path
