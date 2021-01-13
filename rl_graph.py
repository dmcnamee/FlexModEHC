"""RL functions to apply to task graphs.
"""
import numpy as np

import os
import check_inputs as check
import utils


def successor_rep(transmat, discount):
  """Compute successor representation matrix analytically, using the following.

  M = sum_(t=0)^infinity (discount^t T^t) = (I - discount*T)^(-1)  (eq. 7)

  Args:
    transmat: [n_state x n_state] transition matrix, where transmat[i, j] is
      equal to the probability of transitioning from state i from state j
    discount: scalar discount factor between [0 inclusive, 1 exclusive)

  Returns:
    srmat: successor representation matrix M, where srmat[i, j] is equal to the
      expected discounted number of visitations to state j starting from state
      i (eq. 3)

  Raises:
    ValueError: if transmat is not square matrix, has negative entries, or has
      rows that don't sum to 0 or 1; if discount is not a scalar or is not in
      [0, 1) range.
  """
  # check inputs
  check.discount(discount)
  check.transmat(transmat)
  transmat = np.array(transmat, dtype=np.float32)
  n_state = transmat.shape[0]
  srmat = np.linalg.inv(np.eye(n_state) - discount * transmat)
  check.successor_repmat(srmat)
  return srmat


def successor_rep_sum(transmat, discount, threshold=check.TOL):
  """Compute successor representation matrix with sum.

  M_approx = sum_(t=0)^n (discount^t T^t)
  where n is chosen such that max(M - M_approx) <= threshold

  In some cases might be faster than successor_rep().

  Args:
    transmat: [n_state x n_state] transition matrix, where transmat[i, j] is
      equal to the probability of transitioning from state i from state j
    discount: scalar discount factor between [0 inclusive, 1 exclusive)
    threshold: scalar, upper bound for the error in the approximate srmat

  Returns:
    srmat: successor representation matrix M, where srmat[i, j] is equal to the
      expected discounted number of visitations to state j starting from state
      i (eq. 3).

  Raises:
    ValueError: if transmat is not square matrix, has negative entries, or has
      rows that don't sum to 0 or 1; if discount is not a scalar or is not in
      [0, 1) range.
  """
  # check inputs
  check.discount(discount)
  check.transmat(transmat)
  n_state = transmat.shape[0]

  srmat = np.zeros((n_state, n_state))
  transmat_exp = np.eye(n_state)
  discount_exp = 1.
  keep_going = True
  while keep_going:
    add_to_srmat = (discount_exp * discount) * np.dot(transmat_exp, transmat)
    srmat += add_to_srmat
    if np.max(add_to_srmat) <= threshold:
      keep_going = False

  check.successor_repmat(srmat)
  return srmat


def successor_rep_td(transmat, discount, learning_rate, random_state, t_episode,
                     n_episode=1, starting_state=None, srmat0=None,
                     snapshot_times=()):
  """Compute successor representation matrix using TD learning.

  Args:
    transmat: [n_state x n_state] transition matrix, where transmat[i, j] is
      equal to the probability of transitioning from state i from state j
    discount: scalar discount factor between [0 inclusive, 1 exclusive)
    learning_rate: scales the state prediction error used to update the
      successor representation. if higher, learning is faster but more variable.
      Range = [0 (no learning), 1 (complete replacement of previous entry)]
    random_state: random state object, e.g. np.random.RandomState(seed_integer))
      Permits reproducibly
    t_episode: maximum number of timesteps in episode (if there is an absorbing
      state, episode may terminate earlier)
    n_episode: number of episodes (default=1)
    starting_state: scalar index specifying starting state (default=None, in
      which case a state is picked uniform randomly)
    srmat0: optional initialized successor representation matrix (default=None,
      in which case identity matrix np.eye(n_state) is used)
    snapshot_times: iterable of timesteps at which to record successor
      representation during learning so that learning dynamics can be observed.
      for some t_episode, snapshot_times[i] = T will specify episode
      floor(T / t_episode) and t_in_episode T % t_episode (default=())

  Returns:
    srmat: successor representation matrix M, where srmat[i, j] is equal to the
      expected discounted number of visitations to state j starting from state
      i (eq. 3)
    srmat_snapshots: dict of srmats captured at times specified by
      snapshot_times

  Raises:
    ValueError: if transmat is not square matrix, has negative entries, or has
      rows that don't sum to 0 or 1; if discount is not a scalar or is not in
      [0, 1) range.
  """
  # check inputs
  check.discount(discount)
  check.transmat(transmat)
  n_state = transmat.shape[0]
  check.learning_rate(learning_rate)
  check.is_integer_gteq0(t_episode, 't_episode')
  check.is_integer_gteq0(n_episode, 'n_episode')
  if starting_state is not None:
    check.is_integer_gteq0_ltn(starting_state, n_state, 'starting_state')
  if not np.iterable(snapshot_times):
    raise ValueError('snapshot_times must be iterable (list, tuple, or array)')
  if srmat0 is None: srmat0 = np.eye(n_state)
  check.successor_repmat(srmat0)

  t_elapsed = 0
  srmat_snapshots = {}
  srmat = srmat0.copy()
  for _ in range(n_episode):
    # draw a starting state or use provided
    if starting_state is None:
      state_ind = random_state.randint(n_state)
    else: state_ind = starting_state
    state_vec = np.zeros(n_state)
    state_vec[state_ind] = 1.

    for _ in range(t_episode):
      # draw a state
      if np.all(np.abs(transmat[state_ind, :]) < check.TOL):
        break  # if transmat[state_ind] all 0, absorbing state reached.

      state_vec_next = random_state.multinomial(1, transmat[state_ind, :])
      state_ind_next = np.where(state_vec_next)[0][0]

      # compute successor prediction error: state observed + discount * SR at
      # next state, minus previous estimate for SR
      prediction_err = (state_vec + discount * srmat[state_ind_next, :] -
                        srmat[state_ind, :])

      # update with prediction error
      srmat[state_ind, :] = srmat[state_ind, :] + learning_rate * prediction_err

      # record srmat if t_elapsed is in snapshot_times
      if t_elapsed in snapshot_times:
        srmat_snapshots[t_elapsed] = srmat.copy()

      # update state_ind, state_vec
      state_ind = state_ind_next
      state_vec = state_vec_next

      t_elapsed += 1

  return srmat, srmat_snapshots


def optimal_policy(adjmat, reward, discount, beta_sftmx):
  """Compute optimal policy using policy iteration.

  Args:
    adjmat: [n_state x n_state] adjacency matrix, where adjmat[i, j] is
      1 if a transition from state i to j is possible and 0 otherwise
    reward: (n_state,) reward vector, where reward[i] is equal to the reward
      at state i
    discount: scalar discount factor between [0 inclusive, 1 exclusive)
    beta_sftmx: softmax inverse temperature parameter. Range [0, inf). Modulates
      the extent to which valuable actions are prioritized over random ones.
      Under the softmax optimal policy, a next state will be selected with
      probability proportional to the softmax of the value of that state
      compared to adjacent states.
        beta = 0: policy will be uniform over adjacent states at each node, no
          preference given to valuable states
        beta -> infinity: optimal policy will deterministically sample the most
          valuable state adjacent to each node (or uniformly between the most
          valuable states if there are ties). Less valuable states will never
          be selected.

  Returns:
    opt_transmat: the transition matrix under the softmax optimal policy,
      meaning that each action is selected with probability proportional to the
      softmax value under the deterministic optimal policy.
    value:
    srmat: successor representation matrix, where srmat[i, j] is equal to the
      expected discounted number of visitations to state j starting from state
      i (eq. 3)

  Raises:
    ValueError: if transmat is not square matrix, has negative entries, or has
      rows that don't sum to 0 or 1; if discount is not a scalar or is not in
      [0, 1) range.

  """
  # check inputs
  check.unweighted_adjmat(adjmat)
  n_state = adjmat.shape[0]
  check.reward(reward, n_state)
  check.discount(discount)
  check.beta_softmax(beta_sftmx)

  # value iteration
  transmat = utils.l1_normalize_rows(adjmat.copy())  # initialize to random walk
  value = np.dot(successor_rep(transmat, discount), reward)

  keep_going = True
  while keep_going:
    value_last = value.copy()
    value_sftmx = utils.softmax(value, beta=beta_sftmx)

    # Policy Evaluation
    # softmax optimal policy
    transmat = utils.l1_normalize_rows(np.dot(adjmat, np.diag(value_sftmx)))
    value = np.dot(successor_rep(transmat, discount), reward)

    if np.sum(np.abs(value - value_last)) < utils.TOL:  # stop when val converges
        keep_going = False

  # v_sftmx[i] = exp(-beta * v[i]) / sum_i(exp(-beta * v[i]))
  value_sftmx = utils.softmax(value, beta=beta_sftmx)
  # opt_transmat proportional to softmax value of states adj to i
  opt_transmat = utils.l1_normalize_rows(np.dot(adjmat, np.diag(value_sftmx)))

  return opt_transmat, value
