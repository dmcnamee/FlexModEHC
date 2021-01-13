"""Functions to check if variables supplied are valid.
"""
import numpy as np


TOL = 1e-10


def is_scalar_0inc_1exc(scalar, scalar_name):
  """Checks if scalar is scalar and 0<=scalar<1, else raises ValueError."""
  err_msg = ('{} {} invalid: must be scalar such that 0 <= '
             '{} < 1.').format(scalar_name, scalar, scalar_name)
  if not np.isscalar(scalar):
    raise ValueError(err_msg)
  if (scalar < 0.) or (scalar >= 1.):
    raise ValueError(err_msg)


def is_scalar_gteq0(scalar, scalar_name):
  """Checks if scalar is scalar and 0<=scalar, else raises ValueError."""
  err_msg = ('{} {} invalid: must be scalar >= 0.').format(scalar_name, scalar)
  if not np.isscalar(scalar):
    raise ValueError(err_msg)
  if scalar < 0.:
    raise ValueError(err_msg)


def is_integer_gteq0(integer, int_name):
  """Checks if input is integer >= 0, else raises ValueError."""
  err_msg = '{} {} invalid: must be integer >= 0.'.format(int_name, integer)
  if not np.isscalar(integer):
    raise ValueError(err_msg)
  if not isinstance(integer, int):
    raise ValueError(err_msg)
  if integer < 0:
    raise ValueError(err_msg)


def is_integer_gteq0_ltn(integer, n, int_name):
  """Checks if input is integer where 0 <= int < n, else raises ValueError."""
  err_msg = '{} {} invalid: must be integer >= 0, < n.'.format(int_name,
                                                               integer)
  if not np.isscalar(integer):
    raise ValueError(err_msg)
  if not isinstance(integer, int):
    raise ValueError(err_msg)
  if integer < 0:
    raise ValueError(err_msg)
  if integer >= n:
    raise ValueError(err_msg)


def is_nonneg(matrix, matrix_name):
  """Checks all entries in matrix are nonnegaive, else raises ValueError."""
  if np.any(matrix) < 0:
    raise ValueError('{} cannot have negative entries.'.format(matrix_name))


def is_square_matrix(matrix, matrix_name):
  """Checks if matrix is square, else raises ValueError."""
  err_msg = ('{} with shape {} invalid: must be matrix with same'
             ' number of rows and columns.').format(matrix_name, matrix.shape)
  if len(matrix.shape) != 2:
    raise ValueError(err_msg)
  if matrix.shape[0] != matrix.shape[1]:
    raise ValueError(err_msg)


def discount(discount_):
  """Checks if discount is scalar and 0<=discount<1, else raises ValueError."""
  is_scalar_0inc_1exc(discount_, 'discount')


def learning_rate(learning_rate_):
  """Checks if lr is scalar and 0<=lr<1, else raises ValueError."""
  is_scalar_0inc_1exc(learning_rate_, 'learning_rate')


def beta_softmax(beta_):
  """Checks if beta is scalar and 0<=beta, else raises ValueError."""
  is_scalar_gteq0(beta_, 'beta_softmax')


def reward(reward_, n_state):
  """Checks if reward is valid reward vector, else raises ValueError."""
  err_msg = 'reward must be array with shape (n_state,).'
  if reward_.shape != (n_state,):
    raise ValueError(err_msg)


def transmat(transmat_):
  """Checks if transmat is valid transition matrix, else raises ValueError."""
  is_square_matrix(transmat_, 'transmat')
  is_nonneg(transmat_, 'transmat')
  if not np.all([np.min(np.abs(i - np.array([0, 1]))) < TOL
                 for i in np.sum(transmat_, axis=1)]):
    raise ValueError(('All rows of transmat must sum to 1 or 0. You may want to'
                      ' call util.l1_normalize_rows() on the input matrix.'))


def weighted_adjmat(adjmat_):
  """Checks if adjmat is valid adjacency matrix, else raises ValueError."""
  is_square_matrix(adjmat_, 'adjmat')
  is_nonneg(adjmat_, 'adjmat')


def unweighted_adjmat(adjmat_):
  """Checks if adjmat is valid adjacency matrix, else raises ValueError."""
  weighted_adjmat(adjmat_)
  if not np.all([np.min(np.abs(i - np.array([0, 1]))) < TOL
                 for i in adjmat_.reshape(-1)]):
    raise ValueError('All entries in adjmat must be 1 or 0.')


def successor_repmat(srmat_):
  """Checks if srmat is valid successor rep matrix, else raises ValueError."""
  is_square_matrix(srmat_, 'srmat')
  is_nonneg(srmat_, 'srmat')

