#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import config

import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from copy import deepcopy
from scipy.stats import sem
from scipy.signal import correlate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

from simulators import Simulator
from visualization import color_time_covered, save_figure
from numba import jit, prange
from timer import timeit_debug, timeit_info


class Explorer(Simulator):
    """
    Simulator class with additional search analytics.
    INPUTS: inherits from Simulator
            diagnostics = True, computes sampling diagnostics e.g. search efficiency / coverage
    NOTES: serves to model for exploration and sampling/planning objectives
    """
    @timeit_info
    def __init__(self, PROP, rho_init=None, mass=1, no_dwell=True, label='EXPLORER', **kwargs):
        super().__init__(PROP=PROP, rho_init=rho_init, mass=mass, no_dwell=no_dwell, label=label, **kwargs)

    @timeit_debug
    def _set_file_names(self):
        super(Explorer, self)._set_file_names()
        self.fname_objs = os.path.join(self.output_dir, 'objs_' + self.fname_base + '.csv')

    @timeit_debug
    def _set_output_container(self):
        """
        FUNCTION: Establishes pandas dataframes for storing results.
        NOTES: output_simulation records scalar simulation characteristics for every trajectory sample/step
               output_diagnostics records scalar exploration diagnostics for every step
        """
        super()._set_output_container()
        self.fname_sim = os.path.join(self.output_dir, 'sim_' + self.fname_base + '.csv')
        self.fname_diag = os.path.join(self.output_dir, 'diag_' + self.fname_base + '.csv')

        simIX = pd.MultiIndex.from_product([self.ix_samps, self.ix_steps], names=['sample', 'step'])
        self.output_simulation = pd.DataFrame(data=np.nan, index=simIX, columns=['sq_disp', 'jump_length'], dtype='float')
        diagIX = pd.Index(data=self.ix_steps, name='step')
        self.output_diagnostics = pd.DataFrame(data=np.nan, index=diagIX, columns=['time', 'sq_disp_mean', 'sq_disp_std', 'efficiency'], dtype='float')
        self.output_diagnostics.loc[:,'time'] = self.samp_times
        self.diagnostics_computed = False

    @timeit_debug
    def _record_sim_var(self, samp, step=None, **kwargs):
        """Record simulation variable in self.output_simulation"""
        if step is None:
            ix = self.ix_slice[samp,:]
        else:
            ix = self.ix_slice[samp,step]
        for key, value in kwargs.items():
            self.output_simulation.loc[ix,key] = value

    @timeit_debug
    def _retrieve_sim_var(self, samp, step=None, key='sq_disp'):
        """
        FUNCTION: Retrieve simulation variable in self.output_simulation
        INPUT: samp = trajectory sample number
               step = step number, step=None returns entire trajectory
               key  = column in self.output_simulation to return
        """
        if samp is None:
            if step is None:
                ix = self.ix_slice[:,:]
            else:
                ix = self.ix_slice[:,step]
        else:
            if step is None:
                ix = self.ix_slice[samp,:]
            else:
                ix = self.ix_slice[samp,step]
        return self.output_simulation.loc[ix,key]

    @timeit_debug
    def _record_diag_var(self, step=None, **kwargs):
        """Record simulation variable in self.output_diagnostics"""
        if step is None:
            ix = self.ix_slice[:]
        else:
            ix = step
        for key, value in kwargs.items():
            self.output_diagnostics.loc[ix,key] = value

    @timeit_debug
    def _retrieve_diag_var(self, step, key='time'):
        """
        FUNCTION: Retrieve simulation variable in self.output_diagnostics
        INPUT: step = step number
               key  = column in self.output_diagnostics to return
        """
        return self.output_diagnostics.loc[step,key]

    @timeit_info
    def compute_diagnostics(self, target_coverage=0.8, embedded_space=True, flight_vision=False):
        """
        FUNCTION: computes diagnostics of state-space sequence samples generated.
                  including MSD, autocorrelation, covering percentage vs time.
        INPUTS: flight_vision   = states visited during Levy jump
                target_coverage = coverage fraction at which state-space is deemed to be "covered"
                embedded_space  = True, state-space embedded in euclidean domain in which distance is computed
        NOTES: spatial scale parameter to be incorporated.
        """
        # MSD and jump lengths
        self.displacement_measures(embedded_space=embedded_space)
        # search efficiency (incl coverage)
        self.exploration_efficiency(target_coverage=target_coverage, flight_vision=flight_vision)
        self.path_efficiency()
        self.diagnostics_computed = True


    @timeit_info
    def displacement_measures(self, tau_norm=False, remove_offset=False, embedded_space=True):
        """
        FUNCTION: computes displacement measures e.g. jump lengths, MSD.
        INPUTS: tau_norm        = normalize distance measure by time constant (thus distance in units travelled per unit time)
                remove_offset   = remove intercept on log-log plot
                embedded_space  = True, state-space embedded in euclidean domain in which distance is computed
        NOTES: spatial scale parameter to be incorporated.
        """
        # jump lengths
        for samp in range(self.n_samp):
            if embedded_space:
                coords = self._retrieve_state(samp=samp, step=None, coords=True)
                jump_length = compute_jump_length(coords, time_step=1, metric='euclidean')
                jump_length = np.insert(jump_length,0,0) # filler for zero step "jump"
            else:
                states = self._retrieve_state(samp=samp, step=None, coords=False)
                n_traj_step = states.size
                jump_length = np.zeros(n_traj_step)
                for ix, state in enumerate(states[:-1]):
                    jump_length[ix+1] = self.ENV.distance(state, states[ix+1])
                if tau_norm:
                    jump_length = jump_length/np.sqrt(self.PROP.tau)
            self._record_sim_var(samp=samp, step=None, jump_length=jump_length)
        self.jump_length = self._retrieve_sim_var(samp=None, step=None, key='jump_length').unstack().values

        # cumulative displacement
        sq_disp = (self.jump_length**2).cumsum(1)
        if remove_offset:
            sq_disp = sq_disp/np.mean(sq_disp[:,1])
        for samp in range(self.n_samp):
            self._record_sim_var(samp=samp, step=None, sq_disp=sq_disp[samp,:])
        self.sq_disp = self._retrieve_sim_var(samp=None, step=None, key='sq_disp').unstack().values

        # summary stats
        self.msd = np.mean(self.sq_disp, axis=0)
        self.sq_disp_std = np.std(self.sq_disp, axis=0)
        self._record_diag_var(step=None, msd=self.msd)
        self._record_diag_var(step=None, sq_disp_std=self.sq_disp_std)


    @timeit_info
    def occupation_stats(self, flight_vision=True, target_coverage=0.95, paths=True):
        """
        FUNCTION: Computes distinct states visited and coverage fraction vs time (across propagation samples).
        INPUTS: flight_vision   = True, implies that self.coverage refers to visited states (otherwise sampled only)
                target_coverage = coverage fraction at which state-space is deemed to be "covered"
                paths           = operate over filled-in and smoothed sequences (i.e. "paths")
        OUTPUTS: self.coverage = n_step-vector of mean coverage across trajectories per timestep.
        NOTES: operates in parallel across samples
               initial state is counted as visited/sampled
               "samples" refer to states sampled by propagator
               "visits" refer to states occupied in traversal between sampled states
        """
        # n_samp x n_seq_steps matrix of sequence samples
        state_seqs = self._retrieve_state(samp=None, step=None, coords=False)
        if state_seqs.ndim == 1:
            state_seqs = state_seqs.reshape((1,-1))
        self.target_coverage = target_coverage
        self.flight_vision = flight_vision
        self.n_distinct_samples = np.zeros((self.n_seq_steps,))
        self.n_distinct_visits = np.zeros((self.n_seq_steps,))
        self.n_distinct_samples_persamp = np.zeros((self.n_samp, self.n_seq_steps))
        self.n_distinct_visits_persamp = np.zeros((self.n_samp, self.n_seq_steps))

        # count unique samples (without flight vision)
        for ns in range(self.n_seq_steps):
            cov_states = np.isin(self.ix_states, state_seqs[:,:ns+1])
            self.n_distinct_samples[ns] = cov_states.sum()
            for s in range(self.n_samp):
                cov_states = np.isin(self.ix_states, state_seqs[s,:ns+1])
                self.n_distinct_samples_persamp[s,ns] = cov_states.sum()

        # count unique samples (flight vision = agent must visit intermediate states along jump)
        # serial
        # count initial state
        for samp in range(self.n_samp):
            vis_states_samp = state_seqs[samp,0].tolist()
            self.n_distinct_visits_persamp[0] = np.isin(self.ix_states, vis_states_samp).sum()
        for samp in range(self.n_samp):
            vis_states_samp = [state_seqs[samp,0]]
            for ns in range(1,self.n_seq_steps):
                # iterate over samps - integrate over sequences in parallel
                state_start = state_seqs[samp,ns-1]
                state_end = state_seqs[samp,ns]
                # get starts along shortest path
                if (state_start in self.ix_states) and (state_end in self.ix_states):
                    path = nx.shortest_path(self.ENV.G, source=state_start, target=state_end)
                    path = path[1:] # remove start state_start
                    vis_states_samp = np.unique(vis_states_samp + path).tolist()
                self.n_distinct_visits_persamp[samp, ns] = np.isin(self.ix_states, vis_states_samp).sum()

        # parallelized
        # count initial state
        vis_states = state_seqs[:,0].tolist()
        self.n_distinct_visits[0] = np.isin(self.ix_states, vis_states).sum()
        for ns in range(1,self.n_seq_steps):
            # iterate over samps - integrate over sequences in parallel
            for samp in range(self.n_samp):
                state_start = state_seqs[samp,ns-1]
                state_end = state_seqs[samp,ns]
                # get starts along shortest path
                if (state_start in self.ix_states) and (state_end in self.ix_states):
                    path = nx.shortest_path(self.ENV.G, source=state_start, target=state_end)
                    path = path[1:] # remove start state_start
                    vis_states = np.unique(vis_states + path).tolist()
            self.n_distinct_visits[ns] = np.isin(self.ix_states, vis_states).sum() # check distinct visits

        # coverage fraction including all visited states along Levy jump
        self.coverage_visits = self.n_distinct_visits/float(self.n_state) # fraction of total
        self.coverage_visits_persamp = self.n_distinct_visits_persamp/float(self.n_state) # fraction of total
        # coverage fraction based on states sampled in propagator
        self.coverage_samples = self.n_distinct_samples/float(self.n_state) # fraction of total
        self.coverage_samples_persamp = self.n_distinct_samples_persamp/float(self.n_state) # fraction of total
        if flight_vision:
            self.coverage = self.coverage_visits # default
            self.coverage_persamp = self.coverage_visits_persamp # default
        else:
            self.coverage = self.coverage_samples
            self.coverage_persamp = self.coverage_samples_persamp

        if np.any(self.coverage>target_coverage):
            self.time_covered_idx = np.where(self.coverage>target_coverage)[0][0]
            self.time_covered = self.samp_times[self.time_covered_idx]
        else:
            self.time_covered = None
            self.time_covered_idx = None


    @timeit_info
    def exploration_efficiency(self, normalized=False, target_coverage=0.9, time_cost=0., flight_vision=False):
        """
        FUNCTION: Computes the ratio of states visited (or fraction thereof) to distance travelled.
        INPUTS: flight_vision = whether or not states can be observed during multi-state jumps (i.e. Levy flights)
                normalized = normalize efficiency measure with respect to total number of states
                target_coverage = desired coverage fraction
                time_cost = "distance" cost of of sample time (penalizes time ito distance)
        OUTPUTS: self.step_cost = per-step jump length + time_cost
                 self.traj_cost = cumulative step cost
                 self.traj_cost_mean = average cumulative step cost across sampled sequences
                 self.traj_cost_sum = total cumulative step cost across sampled sequences
                 self.exp_eff = n_distinct_visits|samples (in parallel across samples) / traj_cost_mean (average across samples)
        NOTES: computed across sampled trajectories i.e. parallel search
        REFS: Viswanathan et al (1999)
        """
        # compute cumulative trajectory cost
        self.step_cost = deepcopy(self.jump_length)
        self.step_cost += 1
        self.step_cost += time_cost
        self.traj_cost = self.step_cost.cumsum(axis=1)
        self.traj_cost_mean = self.traj_cost.mean(axis=0)
        self.traj_cost_sum = self.traj_cost.sum(axis=0)

        # compute visit/sampling statistics
        self.occupation_stats(target_coverage=target_coverage, flight_vision=flight_vision)

        # combine to compute exploration efficiency
        if normalized:
            if flight_vision:
                self.exp_eff = self.coverage_visits/self.traj_cost_mean
            else:
                self.exp_eff = self.coverage_samples/self.traj_cost_mean
        else:
            if flight_vision:
                self.exp_eff = self.n_distinct_visits/self.traj_cost_sum
            else:
                self.exp_eff = self.n_distinct_samples/self.traj_cost_sum
        self.exp_eff[0] = 1. # convention

    @timeit_info
    def compute_sample_density(self, sigma=0.):
        """
        FUNCTION: computes state density sampled under sequence generation
        INPUTS: sigma = gaussian filter std
        """
        from scipy.ndimage import gaussian_filter

        state_seqs = self._retrieve_state(samp=None, step=None, coords=False)
        unique, counts = np.unique(ar=state_seqs.flatten(), return_counts=True)
        self.state_seqs_density = np.zeros((self.n_state,))
        self.state_seqs_density[unique.astype('int')] = counts/counts.sum()
        if sigma != 0.:
            self.state_seqs_density = gaussian_filter(self.state_seqs_density, sigma=sigma)

    @timeit_info
    def plot_sample_density(self, cmap=plt.cm.hot, cbar=False, vmin=None, vmax=None):
        """
        FUNCTION: plots density of sampled states across sequences
        INPUTS: state_seqs
        OUTPUTS: self.state_seqs_density
        """
        if not hasattr(self, 'state_seqs_density'):
            self.compute_sample_density()

        if self.no_target_axis:
            self.fig = plt.figure(figsize=(10,10))
            self.ax = plt.gca()

        if vmin is None:
            vmin = self.state_seqs_density.min()
        if vmax is None:
            vmax = self.state_seqs_density.max()

        self.ax = self.ENV.plot_state_func(state_vals=self.state_seqs_density, ax=self.ax, cbar=cbar, cmap=cmap, vmin=vmin, vmax=vmax, mask_color='white')
        if self.no_target_axis:
            self.ax.set_title(self.label+', sampling density')


    @timeit_info
    def plot_visit_density(self, cmap=plt.cm.hot, cbar=False, vmin=None, vmax=None):
        """
        FUNCTION: plots density of visited states across sequences
        INPUTS: self.paths_array
        OUTPUTS: self.paths_density
        """
        # compute density
        unique, counts = np.unique(ar=self.paths_array.flatten(), return_counts=True)
        self.paths_density = np.zeros((self.n_state,))
        self.paths_density[unique.astype('int')] = counts/counts.sum()

        if self.no_target_axis:
            self.fig = plt.figure(figsize=(10,10))
            self.ax = plt.gca()

        if vmin is None:
            vmin = self.paths_density.min()
        if vmax is None:
            vmax = self.paths_density.max()

        self.ax = self.ENV.plot_state_func(state_vals=self.paths_density, ax=self.ax, cbar=cbar, cmap=cmap, vmin=vmin, vmax=vmax, mask_color='white')
        if self.no_target_axis:
            self.ax.set_title(self.label+', visit density')



    @timeit_info
    def spike_separation(self):
        """
        FUNCTION: computes time between spikes in sampled sequences as a function of distance between place cells
        NOTES: inspired by Karlsson & Frank (2009), Suh et al (2013)
        """
        samps = self._retrieve_state(samp=None, step=None, coords=False) # n_samps x n_steps
        for samp in range(self.n_samp):
            for s1 in range(self.n_state):
                for s2 in range(s1,self.n_state):
                    # get all #timesteps between s1 and s2
                    print('todo')


    @timeit_info
    def crosscorr_traj(self, interval_size=1., smooth_sigma=0.):
        """
        FUNCTION: cross-correlations estimated from generated trajectories as a function of spatial distance
        NOTES: inspired by Karlsson & Frank, Nat Neuro (2009)
        Pair-wise reactivation in the rest box. Each plot shows rows
        representing the normalized cross-correlegrams
        between all pairs of simultaneously recorded
        neurons with place fields in environment 1, with
        the vertical location of each row being determined by the distance between the two cells place field peaks
        """
        samps = self._retrieve_state(samp=None, step=None, coords=False) # n_samps x n_steps
        time_disp = np.arange(-self.n_step, self.n_step+1)
        ix_tuples = []
        samp_tuples = []
        for s1 in range(self.n_state):
            s1_spikes = (samps==s1).astype('int')
            for s2 in range(s1,self.n_state):
                s2_spikes = (samps==s2).astype('int')
                ccorr = np.array([correlate(s1_spikes[i, :], s2_spikes[i, :], 'full') for i in range(self.n_samp)])
                # normalize within each sample
                ccorr_max = ccorr.max(1); ccorr[ccorr_max!=0,:] = ccorr[ccorr_max!=0,:]/ccorr_max[ccorr_max!=0,None]
                # average over trajectory events
                ccorr = ccorr.mean(0)
                dist = self.ENV.distance(state1=s1, state2=s2, interval_size=interval_size)
                ix_tuples = ix_tuples + [ix for ix in product([s1],[s2],time_disp)]
                samp_tuples = samp_tuples + [i for i in zip(ccorr, [dist for i in range(len(time_disp))])]
        ccorrs_traj = pd.DataFrame(samp_tuples, index=pd.MultiIndex.from_tuples(ix_tuples, names=['state1', 'state2', 'time_displacement']), columns=['ccorr', 'dist'])
        self.ccorrs = ccorrs_traj.reset_index().groupby(['time_displacement','dist']).mean().drop(columns=['state1','state2']).unstack('time_displacement')
        # normalize within each distance
        ccorr_max = self.ccorrs.values.max(1); self.ccorrs.values[ccorr_max!=0,:] = self.ccorrs.values[ccorr_max!=0,:]/ccorr_max[ccorr_max!=0,None]
        if smooth_sigma != 0.:
            self.ccorrs.loc[:,:] = gaussian_filter1d(self.ccorrs, axis=1, sigma=smooth_sigma)
        self.dist = self.ccorrs.index.get_level_values('dist')
        self.time_disp = self.ccorrs.columns.get_level_values('time_displacement')


    @timeit_info
    def crosscorr_rho(self, interval_size=1., smooth_sigma=0.):
        """
        FUNCTION: propagated density cross-correlations as a function of spatial distance
        NOTES: inspired by Karlsson & Frank, Nat Neuro (2009)
        """
        samps = self._retrieve_vector(key='rho_stop') # n_samps x n_steps x n_states
        time_disp = np.arange(-self.n_step, self.n_step+1)
        ix_tuples = []
        samp_tuples = []
        for s1 in range(self.n_state):
            s1_rho = samps[pd.IndexSlice[:,:,s1]].squeeze() # n_samps x n_steps
            for s2 in range(s1,self.n_state):
                s2_rho = samps[pd.IndexSlice[:,:,s2]].squeeze()
                ccorr = np.array([correlate(s1_rho[i, :], s2_rho[i, :], 'full') for i in range(self.n_samp)])
                if smooth_sigma != 0.:
                    ccorr = gaussian_filter1d(ccorr, axis=1, sigma=smooth_sigma)
                ccorr = ccorr.mean(0) # average over trajectory events
                dist = self.ENV.distance(state1=s1, state2=s2, interval_size=interval_size)
                ix_tuples = ix_tuples + [ix for ix in product([s1],[s2],time_disp)]
                samp_tuples = samp_tuples + [i for i in zip(ccorr, [dist for i in range(len(time_disp))])]
        ccorrs_rho = pd.DataFrame(samp_tuples, index=pd.MultiIndex.from_tuples(ix_tuples, names=['state1', 'state2', 'time_displacement']), columns=['ccorr', 'dist'])
        self.ccorrs = ccorrs_rho.reset_index().groupby(['time_displacement','dist']).mean().drop(columns=['state1','state2']).unstack('time_displacement')
        # row normalize
        ccorr_max = self.ccorrs.values.max(1)
        self.ccorrs.values[ccorr_max!=0,:] = self.ccorrs.values[ccorr_max!=0,:]/ccorr_max[ccorr_max!=0,None] # normalize within each distance
        self.dist = self.ccorrs.index.get_level_values('dist')
        self.time_disp = self.ccorrs.columns.get_level_values('time_displacement')

    @timeit_info
    def crosscorr_rho_vectorized(self, interval_size=1.):
        """
        FUNCTION: propagated density cross-correlations as a function of spatial distance
        NOTES: inspired by Karlsson & Frank, Nat Neuro (2009)
        """
        raise ValueError('bugged and slower than looped crosscorr_rho')
        from itertools import product
        samps = self._retrieve_vector(samp=None, step=None) # n_samps x n_steps x n_states
        time_disp = np.arange(-self.n_step, self.n_step+1)
        IX = np.repeat(np.arange(self.n_state).reshape(-1,1),repeats=(self.n_state),axis=1)
        ud_ix = np.triu_indices_from(IX)
        state_tuples = list(zip(IX[ud_ix], IX.T[ud_ix]))
        ix_tuples = [(a,b,c) for ((a,b),c) in product(state_tuples, time_disp)]
        index = pd.MultiIndex.from_tuples(ix_tuples, names=['state1', 'state2', 'time_displacement'])
        self.ccorrs_rho = pd.DataFrame(index=index, columns=['ccorr', 'dist'], dtype='float')

        def ccorr_func(s1,s2):
            s1_rho = samps[pd.IndexSlice[:,:,s1]].unstack().values
            s2_rho = samps[pd.IndexSlice[:,:,s2]].unstack().values
            ccorr = np.array([correlate(s1_rho[i, :], s2_rho[i, :], 'full') for i in range(self.n_samp)])
            ccorr_max = ccorr.max(1)
            ccorr[ccorr_max!=0,:] = ccorr[ccorr_max!=0,:]/ccorr_max[ccorr_max!=0,None] # normalize within each sample
            ccorr = ccorr.mean(0) # average over trajectory events
            return ccorr

        def dist_func(s1,s2):
            return self.ENV.distance(state1=s1, state2=s2, interval_size=interval_size)

        self.ccorrs_rho['ccorr'] = self.ccorrs_rho['ccorr'].unstack('time_displacement').apply(lambda row: ccorr_func(row.name[0],row.name[1]), axis=1, result_type='broadcast')
        self.ccorrs_rho['dist'] = self.ccorrs_rho['dist'].unstack('time_displacement').apply(lambda row: ccorr_func(row.name[0],row.name[1]), axis=1, result_type='broadcast')
        # postproc
        self.ccorrs = self.ccorrs_rho.reset_index().groupby(['time_displacement','dist']).mean().drop(columns=['state1','state2']).unstack('time_displacement')
        # row normalize
        ccorr_max = self.ccorrs.values.max(1)
        self.ccorrs.values[ccorr_max!=0,:] = self.ccorrs.values[ccorr_max!=0,:]/ccorr_max[ccorr_max!=0,None] # normalize within each distance
        self.dist = self.ccorrs.index.get_level_values('dist')
        self.time_disp = self.ccorrs.columns.get_level_values('time_displacement')


    @timeit_info
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def estimate_autocorrelation_sample(self, normalize=True, coords=True):
        """
        FUNCTION: estimates autocorrelation from sampled trajectories
        INPUT:  normalize = True, normalize by variance, False = returns autocovariance
                coords = embed in ambient continuous space
        """
        assert coords, 'graph-based autocorrelations not coded'
        samps = self._retrieve_state(samp=None, step=None, coords=coords) # n_samps x n_steps x n_dim
        # average over sample estimates per lag
        AC = []
        if normalize:
            ix_zero = np.floor(AC.size/2.).astype('int')
            AC = AC/AC[ix_zero] # AC[lag=0] ~= S.var()
        return AC

    @timeit_info
    @jit(nopython=config.jit_nopython, parallel=config.jit_nopython, cache=config.jit_cache)
    def estimate_autocorrelation_fft(self, normalize=True, one_sided=True, coords=True):
        """
        FUNCTION: estimates autocorrelation from sampled trajectories via FFTs
        INPUT:  normalize = True, normalize by variance, False = returns autocovariance
                one_sided = True, average AC over pos/neg lags
                coords = embed in ambient continuous space
        REFS: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        NOTES: results strong correlations which scale with n_step (rather than tau/alpha)
               and artifacts when conbined with subtracting the mean
        """
        assert coords, 'graph-based autocorrelations not coded'
        from scipy.signal import correlate
        from scipy.stats import zscore
        samps = self._retrieve_state(samp=None, step=None, coords=coords) # n_samps x n_steps x n_dim
        ACs = []
        for i in prange(self.n_samp):
            ACd = []
            for d in prange(self.n_dim):
                S = samps[i,:,d]
                AC = correlate(S, S, mode='full', method='auto')
                ACd.append(AC)
            ACs.append(np.array(ACd).mean(0))
        AC = np.array(ACs).mean(0)
        if normalize:
            ix_zero = np.floor(AC.size/2.).astype('int')
            AC = AC/AC[ix_zero] # AC[lag=0] ~= S.var()
        if one_sided:
            AC = AC[np.floor(AC.size/2.).astype('int'):]
        return AC

    def plot_coverage(self, across_samp=False, func_of_time=True, color='k', alpha=0.3):
        """
        FUNCTION: Plots self.coverage
        INPUTS: across_samp = plot mean/sem of coverage across sequence samples
                              (otherwise plot total coverage from all samples)
                func_of_time=True, as a function of time else average distance
                color = line color
                alpha = alpha of standard error fill
        """
        assert self.diagnostics_computed, 'EXPLORER: diagnostics not computed'
        self.fig_axis()

        if func_of_time:
            if across_samp is False:
                self.ax.plot(self.samp_times, self.coverage, '-', label=self.label, color=color)
            else:
                cov_mean = self.coverage_persamp.mean(0)
                cov_sem = sem(self.coverage_persamp, axis=0)
                self.ax.plot(self.samp_times, cov_mean, '-', label=self.label, color=color)
                self.ax.fill_between(self.samp_times, cov_mean-cov_sem, cov_mean+cov_sem, color=color, alpha=alpha)

            # highlight target_coverage
            self.ax.axhline(self.target_coverage, ls='--', color=color_time_covered, zorder=-10)
            if self.time_covered is not None:
                self.ax.scatter(self.time_covered, self.target_coverage, color=color_time_covered, zorder=99)
            self.ax.set_xlabel('time')
        else:
            if across_samp is False:
                self.ax.plot(self.traj_cost_mean, self.coverage, '-', label=self.label, color=color)
            else:
                cov_mean = self.coverage_persamp.mean(0)
                cov_sem = sem(self.coverage_persamp, axis=0)
                self.ax.plot(self.traj_cost_mean, cov_mean, '-', label=self.label, color=color)
                self.ax.fill_between(self.traj_cost_mean, cov_mean-cov_sem, cov_mean+cov_sem, color=color, alpha=alpha)
            self.ax.set_xlabel('average distance traversed')
        self.ax.set_ylabel('coverage')
        self.ax.set_title('state-space coverage')
        self.ax.set_ylim([0,1])
        sb.despine(ax=self.ax, top=True, right=True, left=False, bottom=False)

    def plot_exploration_efficiency(self, color='k'):
        """
        FUNCTION: plots self.exp_eff as a function of time
        """
        assert self.diagnostics_computed, 'EXPLORER: diagnostics not computed'
        self.fig_axis()
        if self.ax is None:
            plt.figure()
            self.ax = plt.gca()
        self.ax.plot(self.samp_times, self.exp_eff, '-', label=self.label, color=color)
        if self.time_covered is not None:
            self.ax.scatter(self.time_covered, self.exp_eff[self.time_covered_idx], color=color_time_covered, zorder=99)
        # self.ax.set_ylim([0,1.05])
        sb.despine(ax=self.ax, top=True, right=True, left=False, bottom=False)
        self.ax.set_xlabel('time')
        self.ax.set_ylabel('coverage / distance traversed')
        self.ax.set_title('state-space search efficiency')
        # self.ax.set_ylim([0,None])
        self.ax.legend()

    def plot_exploration_efficiency_dist(self, color='k'):
        """
        FUNCTION: plots self.exp_eff as a function of distance
        """
        assert self.diagnostics_computed, 'EXPLORER: diagnostics not computed'
        self.fig_axis()
        if self.ax is None:
            plt.figure()
            self.ax = plt.gca()
        self.ax.plot(self.traj_cost_mean, self.exp_eff, '-', label=self.label, color=color)
        sb.despine(ax=self.ax, top=True, right=True, left=False, bottom=False)
        self.ax.set_xlabel('average distance traversed')
        self.ax.set_ylabel('coverage / distance traversed')
        self.ax.set_title('state-space search efficiency')
        # self.ax.set_ylim([0,None])
        self.ax.legend()

    def plot_msd(self, loglog=True, cut_at_sat=True, remove_time0=True):
        """
        FUNCTION: Plots MSD estimate for process.
        INPUTS: self.sq_disp = samples of squared displacements, n_samp x n_state
                cut_at_sat = True, truncates plot where displacements saturate.
                loglog = plot on log-log scale
                remove_time0 = remove zero time displacement from plot (must equal zero)
        """
        assert self.diagnostics_computed, 'EXPLORER: diagnostics not computed'
        self.fig_axis()
        sq_disp = self.sq_disp
        msd = self.msd
        sq_disp_std = self.sq_disp_std
        samp_times = deepcopy(self.samp_times)
        if remove_time0:
            # remove zero displacement (log(0)=-inf
            samp_times = samp_times[1:]
            sq_disp = sq_disp[:,1:]
            msd = msd[1:]
            sq_disp_std = sq_disp_std[1:]

        if self.n_samp < 10:
            for ns in range(self.n_samp):
                self.ax.plot(samp_times,sq_disp[ns,:])
        else:
            base_line, = self.ax.plot(samp_times, msd, label=self.label)
            self.ax.fill_between(samp_times, msd-sq_disp_std, msd+sq_disp_std, interpolate=True, facecolor=base_line.get_color(), alpha=0.2)
            times_rep = np.tile(samp_times,(self.n_samp,1))
            self.ax.scatter(times_rep.flatten(),sq_disp.flatten(), alpha=0.3, s=0.4, zorder=-10)

        if loglog:
            self.ax.set(xscale='log', yscale='log')
        self.ax.set_xlabel('time step')
        self.ax.set_ylabel('mean squared displacement')
        if loglog:
            self.ax.set_xlim([np.log(1),self.ax.get_xlim()[1]])
            # self.ax.set_ylim([np.log(1),self.ax.get_ylim()[1]])
            self.ax.set_ylim([np.log(1),self.ax.get_xlim()[1]])
        else:
            self.ax.set_xlim([0,self.n_state/2])
            self.ax.set_ylim([0,self.ax.get_ylim()[1]])
        self.ax.plot(self.ax.get_xlim(), self.ax.get_xlim(), ls='--', color='k')
        self.ax.set_title('state-space displacement over time')
        self.ax.legend(loc='lower right')

    def plot_diagnostics(self, figdir=None):
        """
        FUNCTION: plots diagnostics of sequences generated.
                Including MSD, autocorrelation, covering percentage vs time.
        INPUTS: self = propagator with propagated sequences.
                figdir = output figure diretory
        """
        assert self.diagnostics_computed, 'EXPLORER: diagnostics not computed'
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18,6), sharey=False)
        self.set_target_axis(ax=self.axes[0])
        self.plot_coverage()
        self.set_target_axis(ax=self.axes[1])
        self.plot_exploration_efficiency()
        self.set_target_axis(ax=self.axes[2])
        self.plot_msd()
        plt.tight_layout()
        if figdir is not None:
            save_figure(fig=self.fig, figdir=figdir, fname_base='exploration_analysis', file_ext='.png')



def msd(X):
    """
    X is a dxt matrix of estimated place-cell encoded positions
    d is the state dimension
    t is the number of sampled timepoints
    """
    t = np.atleast_2d(X).T.shape[0]
    nshifts = np.floor(t/4).astype('int')
    disps = np.zeros((nshifts,t))
    # computes shift number x start time point matrix of displacements
    for n in range(nshifts):
        disps[n,:] = X.shift(periods=-n) - X
    M = np.nanmean(disps**2, axis=1)
    return M


def shortest_path(W, source, target):
    """
    FUNCTION: finds shortest path between two states.
    INPUTS: W = (weighted) adjacency matrix
            source = start state
            target = target state
    """
    import networkx as nx
    return nx.shortest_path(nx.from_numpy_matrix(W), source=source, target=target, weight=None, method='dijkstra')


def GW_distance(world_array, s1, s2=None, metric='euclidean'):
    """
    FUNCTION: Returns the gridworld distance between states s1 and s2.
    OUTPUT: #states-in-seq x #states-in-seq matrix of state distances ordered by time
    NOTES: s1 and s2 can be vectors.
    """
    from utils import GWix2coords
    s1coords = GWix2coords(world_array=world_array,ix=s1)
    if s2 is not None:
        s2coords = GWix2coords(world_array=world_array,ix=s2)
        print('to do')
    else:
        from scipy.spatial.distance import pdist
        return pdist(s1coords, metric=metric)


def squared_displacement(coords, coords_target=None):
    """
    FUNCTION: compute pairwise squared displacements
    NOTES: squared displacements are the cumulative sum of squared "jump_lengths" assuming a euclidean metric
    """
    if coords_target is None:
        coords_target = coords[0,:] # initial coordinates by default
    return np.sum((coords-coords_target)**2, axis=1)


def compute_jump_length(coords, time_step=1, metric='euclidean'):
    """
    FUNCTION: Computes jump lengths between successive steps over trajectories.
    INPUTS: coords      = trajectory of state-space coordinates
            time_step   = time step lag over which to compute jump lengths
    OUTPUTS: jump_length
    """
    dist_mat = squareform(pdist(coords, metric=metric))
    return np.diag(dist_mat, k=time_step)


def squared_disp_derivative(pairwise_dist):
    """
    FUNCTION: Computes mean squared-displacement "derivative".
    INPUTS:
    n_seq_steps x n_seq_steps symmetric distance matrix
    OUTPUTS: n_seq_steps x 1 vector of displacements
    """
    T = pairwise_dist.shape[0]
    M = np.zeros((T,))*np.nan
    for t in range(T):
        M[t] = np.nanmean(np.diag(pairwise_dist,k=t))

    return M
