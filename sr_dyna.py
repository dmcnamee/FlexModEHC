import numpy as np
import rl_graph
import utils
from utils import to_one_hot


def update_reward(reward, state_sequence, reward_observed, alpha):
  """Update reward vector.

  Args:
    reward: length n_state reward vector
    state_sequence: iterable containing sequence of state inds.
    reward_observed: iterable containing sequence of rewards observed (must
      have same length as state_sequence)
    alpha: scalar learning rate between [0 inclusive, 1 inclusive] saying how
      much to update transmat on each timestep

  Returns:
    updated reward vector
  """
  for reward_obs, state_ind in zip(reward_observed, state_sequence):
    if state_ind is not None:
      reward[state_ind] = (1-alpha) * reward[state_ind] + alpha * reward_obs
  return reward


def update_transition_matrix(transmat, sequence, alpha):
  """Update transition matrix.

  Args:
    transmat: n_state x n_state transition matrix (each row must sum to 1)
    sequence: iterable containing sequence of state inds
    alpha: scalar learning rate between [0 inclusive, 1 inclusive] saying how
      much to update transmat on each timestep

  Returns:
    updated transition matrix
  """
  n_states = transmat.shape[0]

  for state_ind, state_ind_next in zip(sequence[:-1], sequence[1:]):
    if (state_ind is not None) and (state_ind_next is not None):
      state_next_vec = to_one_hot(state_ind_next, n_states)
      transmat[state_ind, :] = ((1-alpha) * transmat[state_ind, :] +
                                alpha * state_next_vec)
  return transmat


def update_sr(sr, sequence, discount, learning_rate):
  """Update SR matrix.

  Args:
    sr: n_state x n_state matrix
    sequence: iterable containing sequence of state inds
    discount: scalar discount factor between [0 inclusive, 1 exclusive)
    learning_rate: scalar learning rate between [0 inclusive, 1 inclusive]

  Returns:
    updated sr matrix
  """
  n_states = sr.shape[0]

  for state_ind, state_ind_next in zip(sequence[:-1], sequence[1:]):
    if (state_ind is not None) and (state_ind_next is not None):
      state_vec = to_one_hot(state_ind, n_states)
      # compute successor prediction error:
      # state observed + discount * SR at next state, minus previous estimate
      pred_err = state_vec + discount * sr[state_ind_next, :] - sr[state_ind, :]

      # use prediction error to update
      sr[state_ind, :] = sr[state_ind, :] + learning_rate * pred_err

  return sr
