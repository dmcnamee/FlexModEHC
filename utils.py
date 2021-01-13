#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import config
import numpy as np

from copy import deepcopy
from scipy.special import softmax as softmax_scipy
from sklearn.preprocessing import normalize
from timer import timeit_debug, timeit_info

import check_inputs as check

# SETTINGS
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
diag_actions = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]  # UP-RIGHT, DOWN-RIGHT, UP-LEFT, DOWN-LEFT
nA = len(actions)  # assuming gridworlds
TOL = 0.001

from scipy.sparse import csr_matrix

if config.sparsify:
    from scipy.sparse.linalg import inv as invert_matrix
else:
    from scipy.linalg import inv as invert_matrix


def to_one_hot(ind, n, dtype=np.float32):
    """Convert index ind to onehot of length n."""
    onehot = np.zeros(n, dtype=dtype)
    onehot[ind] = 1.0
    return onehot


def signed_amp(x):
    """Return sign(x) * amp(x), where amp is amplitude of complex number."""
    return np.sign(np.real(x)) * np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2)


def ensure_dir(f):
    """Ensures a directory exists and returns the full path."""
    d = os.path.abspath(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


@timeit_debug
def create_tensor(shape, sparse=config.sparsify, fill_value=0.0):
    if sparse:
        return csr_matrix(shape)
    else:
        return np.ones(shape) * fill_value


@timeit_debug
def identity_matrix(nstates, sparse=config.sparsify):
    I = create_tensor((nstates, nstates))
    I[np.diag_indices(nstates)] = 1.0
    return I


@timeit_debug
def row_norm_minsub(X):
    """L1 normalization and min value subtraction"""
    X = X - X.min()
    # return X/X.sum(axis=1)
    return normalize(X.copy(), norm="l1", axis=1)  # handles zero denominators


@timeit_debug
def row_norm(X):
    """L1 normalization"""
    # return X/X.sum(axis=1)
    return normalize(X.copy(), norm="l1", axis=1)  # handles zero denominators


@timeit_debug
def l1_normalize_rows(mat):
    """Normalize non-zero rows of mat so that they sum to 1.

  Args:
    mat: matrix to normalize

  Returns:
    l1normmat: matrix with rows that sum to 1 or 0.
  """
    denom = np.sum(mat, axis=1)
    denom[denom == 0] = 1.0
    l1normmat = np.divide(mat.T, denom).T
    return l1normmat


@timeit_debug
def normalize_exp(X, beta=1.0):
    """FUNCTION: Boltzmann normalization using exp function"""
    X = beta * X
    if -X.min() > X.max():
        # control for underflow
        b = X.min()
        y = np.exp(X + b)
    else:
        # control for overflow
        b = X.max()
        y = np.exp(X - b)
    return y / y.sum()


@timeit_debug
def normalize_logsumexp(X, beta=1.0):
    """FUNCTION: Boltzmann normalization using logsumexp function.
       NOTES: Good for huge range spanning positive and negative values."""
    from scipy.special import logsumexp

    X = beta * X
    Y = X - X.min() + 1.0
    P = np.exp(np.log(Y) - logsumexp(Y))
    return P


@timeit_debug
def cutoff_exp_underflow(X, cutoff=-100):
    X[X < config.min_val] = 0.0
    return X


@timeit_debug
def norm_density(V, beta=1.0, type="l1"):
    """
    FUNCTION: normalize to [0,1].
    INPUTS: V = values to normalize ("negative energies")
            beta = "inverse temperature" scaling
            type = type of normalization, L1, boltzmann
    """
    V[np.isinf(V)] = 0.0
    # shift into positive range
    # (alpha>1, sometimes results in negative Y values presumably due to precision issues)
    if (V < 0).any():
        V = V - V.min()
    if type == "l1":
        P = V / V.sum()
    elif type == "boltzmann":
        P = normalize_logsumexp(V, beta=beta)
    else:
        raise ValueError("Unknown normalization requested.")
    return P


@timeit_debug
def row_softmax_norm(X, beta=1.0):
    for r in range(X.shape[0]):
        X[r, :] = norm_density(X[r, :], beta=beta, type="boltzmann")
    return X


@timeit_debug
def softmax_policy_transitions(V, A, beta=1.0):
    """
    Transition matrix for a softmax policy from value function V.
    V       = value function
    A       = adjacency matrix
    beta    = inverse temperature (decision noise)
    """
    n_state = A.shape[0]
    TP = np.zeros(A.shape)
    for s1 in range(n_state):
        six = A[s1, :] != 0
        TP[s1, six] = np.exp(beta * V[six])
    if np.any(np.isinf(TP)):
        raise ValueError("Exponential overflow? Beta too high? System too cold?")
    TP = row_norm(TP)
    return TP


@timeit_debug
def graph_laplacian(X):
    from scipy.sparse.csgraph import laplacian

    return laplacian(X, normed=False)


@timeit_debug
def symnorm_graph_laplacian(X):
    from scipy.sparse.csgraph import laplacian

    return laplacian(X, normed=True)


@timeit_debug
def is_symmetric(X):
    return np.all(X.T == X)

@timeit_debug
def is_normal(X):
    """checks if matrix X is normal """
    return np.allclose(X.T.conj().dot(X), X.dot(X.T.conj()))

@timeit_debug
def is_unitary(X):
    """ checks if matrix X is unitary """
    return np.allclose(np.eye(X.shape[0]), X.dot(X.T.conj()))


@timeit_debug
def rescale(V):
    V = V - V.min()
    return V / V.sum()


@timeit_debug
def GWcoords2ix(world_array, coords):
    """
    Returns the state index corresponding to state coordinates in gridworld world_array.
    """
    space_shape = world_array.shape
    IX0, IX1 = np.where(world_array == 0)
    n_state = IX0.size
    S = []
    for si in range(n_state):
        S.append(np.ravel_multi_index((IX0[si], IX1[si]), space_shape))
    sa = np.ravel_multi_index(coords, space_shape)
    return int(np.where(np.array(S) == sa)[0][0])


@timeit_debug
def GWix2coords(world_array, ix):
    """
    Returns the state coords corresponding to state index in gridworld world_array.
    OUTPUT: x,y coordinates
    """
    if type(ix) is not np.ndarray:
        IX0, IX1 = np.where(world_array == 0)
        return IX1[int(ix)], IX0[int(ix)]
    else:
        return GWixvec2coords(world_array, ix)[0].astype("int")


@timeit_debug
def GWixvec2coords(world_array, ix_vector):
    """
    Returns the state coords vector corresponding to state index vector in gridworld world_array.
    """
    ns = ix_vector.shape[0]
    coords = np.zeros((ns, 2))
    for i in range(ns):
        coords[i, :] = GWix2coords(world_array, ix_vector[i])
    return coords.astype("int")


@timeit_debug
def Amat(gridworld, diag_move=False):
    """
    Returns adjacency matrix of gridworld.
    """
    space_shape = gridworld.shape
    IX0, IX1 = np.where(gridworld == 0)
    n_state = IX0.size
    S = []
    for si in range(n_state):
        S.append(np.ravel_multi_index((IX0[si], IX1[si]), space_shape))
    A = np.zeros((n_state, n_state))
    for si in range(n_state):
        x = IX0[si]
        y = IX1[si]
        for D in actions:
            xa = x + D[0]
            ya = y + D[1]
            if gridworld[xa, ya] == 0:  # is open
                sa = np.ravel_multi_index((xa, ya), space_shape)
                sai = np.where(np.array(S) == sa)[0][0]
                A[si, sai] = 1
        if diag_move:
            for D in diag_actions:
                xa = x + D[0]
                ya = y + D[1]
                if gridworld[xa, ya] == 0:  # is open
                    sa = np.ravel_multi_index((xa, ya), space_shape)
                    sai = np.where(np.array(S) == sa)[0][0]
                    A[si, sai] = 1
    return A


@timeit_debug
def Amat2Tact(A, world_array):
    """
    Expresses an adjacency matrix as a tensor T[s,a,snext]
    for the purposes of using policyiteration functions.
    """
    n_state = A.shape[0]
    Tact = np.zeros((n_state, nA, n_state))
    for s1 in range(n_state):
        for s2 in range(n_state):
            if A[s1, s2] > 0:
                s1x, s1y = GWix2coords(world_array, s1)
                s2x, s2y = GWix2coords(world_array, s2)
                ax = s2x - s1x
                ay = s2y - s1y
                ai = actions.index((ax, ay))
                Tact[s1, ai, s2] = A[s1, s2]
    return Tact


@timeit_debug
def Rvec2Ract(R, Tact, boundmult=2):
    """
    Expresses reward vector as a transition tensor.
    boundmult = boundary multiplier of step cost
    """
    n_state = Tact.shape[0]
    nA = Tact.shape[1]
    Ract = np.zeros(Tact.shape)
    for s1 in range(n_state):
        for a in range(nA):
            for s2 in range(n_state):
                if Tact[s1, a, s2] != 0:  # a valid transition
                    Ract[s1, a, s2] = R[s2]
                else:
                    Ract[s1, a, s2] = (
                        boundmult * R.min()
                    )  # remain in place (e.g. off boundary)
    return Ract


@timeit_info
def eigen_decomp(X, real_part=False, right=True, sparse_comp=False):
    """
    FUNCTION: Computes the eigen-decomposition of X.
    INPUTS: X           = square matrix
            real_part   = suppress complex part of evals,evecs
            sparse_comp = sparse matrix computation
            right       = True, right-multiplying transition/generator matrix i.e.
                            dot(rho) = rho O
    NOTE: Eigenvectors are organized column-wise!
          Sparse format typically not faster for eigen-decomposition.
          # TODO check that numpy/scipy is compiled against openblas as this will parallelize eigen_decomp automagically.
    """
    if sparse_comp:
        import scipy.sparse.linalg as LA
        import scipy.sparse as sp

        X = sp.csr_matrix(X)
        if right:
            # right eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigsh(X)
            else:
                evals, EVECS = LA.eigs(X)
        else:
            # left eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigsh(X.T)
            else:
                evals, EVECS = LA.eigs(X.T)
    else:
        import numpy.linalg as LA

        if right:
            # right eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigh(X)
            else:
                evals, EVECS = LA.eig(X)
        else:
            # left eigenvectors
            if is_symmetric(X):
                evals, EVECS = LA.eigh(X.T)
            else:
                evals, EVECS = LA.eig(X.T)

    # eigenspectrum ordering from low-frequency (low abs e-vals) to high-frequency (high abs e-vals)
    ix = np.argsort(np.abs(evals))
    EVECS = EVECS[:, ix]
    evals = evals[ix]
    evals[np.abs(evals) < config.min_val] = 0.0

    if real_part:
        evals = np.real(evals)
        EVECS = np.real(EVECS)
    return evals, EVECS


def eig(x, order="descend", sortby=signed_amp):
    """Computes eigenvectors and returns them in eigenvalue order.

  Args:
    x: square matrix to eigendecompose
    order: 'descend' or 'ascend' to specify in which order to sort eigenvalues
      (default='descend')
    sortby: function transforms a list of (possibly complex, possibly mixed
      sign) into real-valued scalars that can be sorted without ambiguity
      (default=signed_amp)

  Returns:
    evals: matrix with eigenvector columns
    evecs: array of eigenvectors
  """
    assert x.shape[0] == x.shape[1]
    n = x.shape[0]
    evals, evecs = np.linalg.eig(x)

    ind_order = list(range(n))
    ind_order = [x for _, x in sorted(zip(sortby(evals), ind_order))]
    if order == "descend":
        ind_order = ind_order[::-1]
    evals = evals[ind_order]
    evecs = evecs[:, ind_order]
    return evals, evecs


@timeit_info
def process_eigen_grad(egrad, n_state):
    """
    Sets the eigenvector gradient.
    INPUTS:
    egrad = float the evector fraction to consider.
            positive float starts from top (low-frequency) EVECS.
            negative float starts from bottom (high-frequency) EVECS.
    OUTPUT:
    Efactors = vector of evec weights.
    """
    if hasattr(egrad, "__len__"):
        Efactors = egrad
        if Efactors.size < n_state:  #  default is to weight from low-frequency
            Efactors = np.pad(Efactors, (0, n_state - Efactors.size), "constant")
    else:
        Efactors = np.ones((np.floor(n_state * np.abs(egrad)).astype("int"),))
        if Efactors.size < n_state:
            if egrad > 0.0:  # take low-frequency EVECS
                Efactors = np.pad(Efactors, (0, n_state - Efactors.size), "constant")
            else:  # take high-frequency EVECS
                Efactors = np.pad(Efactors, (n_state - Efactors.size, 0), "constant")
    Efactors = Efactors[:n_state]
    return Efactors


def check_commutation(A, B):
    """
    FUNCTION: check if matrices commute.
    """
    thresh = 10 ** -10
    if np.any(np.abs(commutator(A, B)) > thresh):
        print("check_commutation: matrices do not commute.")
        return False
    else:
        print("check_commutation: matrices commute.")
        return True


def commutator(A, B):
    """
    FUNCTION: commutator.
    """
    return np.dot(A, B) - np.dot(B, A)


def simultaneous_diag(A):
    """
    FUNCTION: simultaneous diagonalization of set of matrices.
    """
    print("todo")  # TODO


def SR(T, gamma=0.9):
    """
    Returns the successor representation based on the state transition matrix.
    INPUTS: T = state transition matrix
            gamma = foresight parameter, equivalently discounting (0<gamma<1)
    OUTPUTS: D = successor representation as state-parametrized matrix
    """
    I = identity_matrix(T.shape[0])
    return invert_matrix(I - gamma * T)


@timeit_debug
def SRmat(T, goali=None, gamma=0.99):
    """SR computation with goal absorption"""
    n_state = T.shape[0]
    if goali is None:
        # discounted, infinite-horizon
        Tnorm = row_norm(T)
        return np.linalg.inv(np.eye((n_state, n_state)) - gamma * Tnorm)
    else:
        T[goali, :] = 0  # set to absorb, no discount
        Tnorm = row_norm(T)
        return np.linalg.inv(np.eye((n_state, n_state)) - Tnorm)


@timeit_debug
def real_dtype(V):
    """
    Converts a complex array to real.
    Checks that imag component is neglible.
    """
    if V.imag.sum() < np.finfo(float).eps:
        return V.real
    else:
        raise ValueError("imaginary component of vector is not neglible.")


def gaussian_spheres_custom_dist(pdist_mat, sigma):
    """Evaluate many Gaussian spheres over custom pairwise distance matrix.

  G[i, j] =  1./(sigma*sqrt(2*pi)) * exp(-.5*dist(i, j)^2 / sigma^2)

  where dist(i, j) = pdist_mat[i, j]

  Args:
    pdist_mat: [n_pts x n_pts] matrix of pairwise distances between points
    sigma: scalar width of all Gaussians (same for each Gaussian, same in all
      dimensions)

  Returns:
    gaussians: [n_pts x n_pts] matrix of Gaussian evaluated for the
      corresponding pairwise distance

  Raises:
    ValueError: if sigma not scalar and greater than 0, if any entries in
      pdist_mat are negative
  """
    check.is_scalar_gteq0(sigma, "sigma")
    if np.any(pdist_mat < 0):
        raise ValueError("pdist_mat should not have any negative entries.")
    return (
        1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * pdist_mat ** 2 / sigma ** 2)
    )


def gaussian_spheres(eval_pos, mu_pos, sigma):
    """Evaluate many Euclidean Gaussian spheres at many points in n-D space.

  G[i, j] =  1./(sigma*sqrt(2*pi)) * exp(-.5*dist(i, j)^2 / sigma^2)

  where Euclidean dist(i,j) = sqrt(sum(x^2+y^2+...))

  Args:
    eval_pos: [n_pts x dim] matrix of positions at which to evaluate each
      Gaussian
    mu_pos: [n_gauss x dim] matrix of coordinates of the center of each Gaussian
    sigma: scalar width of all Gaussians (same for each Gaussian, same in all
      dimensions)

  Returns:
    gaussians: [n_pts x n_gauss] matrix of each Gaussian evaluated at each point

  Raises:
    ValueError: if sigma not scalar and greater than 0, if mu_pos.shape[1]
      and eval_pos.shape[1] are different
  """
    # check inputs
    check.is_scalar_gteq0(sigma, "sigma")
    if len(mu_pos.shape) != 2:
        raise ValueError("mu_pos [n_gauss x dims] matrix should have 2 dimensions.")
    if len(mu_pos.shape) != 2:
        raise ValueError("eval_pos [n_pts x dims] matrix should have 2 dimensions.")
    if mu_pos.shape[1] != eval_pos.shape[1]:
        raise ValueError(
            ("mu_pos.shape[1] and eval_pos.shape[1] should be the " "same.")
        )

    n_eval = len(eval_pos)
    n_center = len(mu_pos)
    gaussians = np.zeros((n_eval, n_center))
    for i in range(n_center):
        dist = np.sqrt(np.sum((eval_pos - mu_pos[i]) ** 2, axis=1))
        gaussian = (
            1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * dist ** 2 / sigma ** 2)
        )
        gaussians[:, i] = gaussian.copy()

    return gaussians


def softmax(vec, beta):
    """Apply softmax to vector vec.

  Args:
    vec: vector to compute softmax over
    beta: softmax inverse temperature parameter. Range [0 inclusive, infinity).
      For beta = 0, vec_softmax[i] will be uniform for all entries (1/len(vec)).
      As beta approaches infinity, vec_softmax[i] will approach 1 for
        argmax(vec) and zero everywhere else.

  Returns:
    vec_sftmx: vector of same shape as vec, where
      vec_softmax[i] = exp(beta vec[i]) / sum_i(exp(beta vec[i]))
  """
    check.beta_softmax(beta)
    if beta == np.inf:  # handle separately to avoid numerical instabilities
        vec_sftmx = np.zeros(vec.shape)
        vec_sftmx[vec == np.max(vec)] = 1.0
    else:
        vec_sftmx = softmax_scipy(vec * beta)
        # vec_sftmx = np.exp(vec * beta) # overflows
        # vec_sftmx = vec_sftmx / np.sum(vec_sftmx)  # normalize so sums to 1
    return vec_sftmx


def adjust_range(x, rng):
    """Rescale + recenter x such that min(x), max(x) = rng[0], rng[1].

  Args:
    x: vector or matrix. must contain at least 2 different values to avoid
      dividing by zero.
    rng: [new_min, new_max] range to rescale x

  Returns:
    x_reranged: same dims as x, rescaled to have new min and max specified by
      rng.
      x_reranged = (x-min(x)) / (max(x) - min(x)) * new_max + new_min

  Raises:
    ValueError: if range is not length 2, if x does not contain at least two
      non-identical values
  """
    if len(rng) != 2:
        raise ValueError("rng should have length 2.")
    if len(np.unique(x)) < 2:
        raise ValueError("x must contain at least two non-identical values.")
    new_min, new_max = rng
    return (x - np.min(x)) / np.ptp(x) * new_max + new_min


def check_if_intersect(s1, s2):
    """Check if two segment structs intersect.

  Args:
    s1: segment 1 (struct with fields x1, x2, y1, y2 delimiting segment)
    s2: segment 1 (struct with fields x1, x2, y1, y2 delimiting segment)

  Returns:
    do_intersect: True if segments intersect, False otherwise
  """
    left = max(min(s1.x1, s1.x2), min(s2.x1, s2.x2))
    right = min(max(s1.x1, s1.x2), max(s2.x1, s2.x2))
    bottom = max(min(s1.y1, s1.y2), min(s2.y1, s2.y2))
    top = min(max(s1.y1, s1.y2), max(s2.y1, s2.y2))
    if (left > right) or (bottom > top):  # no overlap
        return False
    elif (left == right) and (bottom == top):  # intersect in point
        return True
    else:  # intersect in segment
        return True


def cart2pol(x, y):
    """Convert from cartesian to polar coordinates (uses radians)."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """Convert from polar to cartesian coordinates (uses radians)."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    INPUTS: xy = npoints x 2 set of coordinates
            radians = to rotate by
            origin = to rotate around

    https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    x = xy[:, 0]
    y = xy[:, 1]
    offset_x, offset_y = origin
    adjusted_x = x - offset_x
    adjusted_y = y - offset_y
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return np.hstack([qx.reshape(-1, 1), qy.reshape(-1, 1)])


class Struct(object):
    """Turn dictionary into struct"""

    def __init__(self, **entries):
        self._dict = entries
        self.__dict__.update(entries)

    def convert2dict(self):
        return self._dict


def struct(d):
    """Call struct conversion."""
    return Struct(**d)


def pos_dict(xymat):
    """Convert from xy matrix to pos_dict object used by networkx."""
    return {i: xymat[i, :] for i in range(xymat.shape[0])}


def transmat_ss_to_sas(T):
    """
    Reshapes a graph transition matrix (SxS) to a FiniteMDP transition matrix with actions (SxAxS).
    NOTES: Unique action for every target state.
    """
    import numpy as np

    n_state = T.shape[0]
    Tsas = np.zeros((n_state, n_state, n_state))
    for s1 in range(n_state):
        for s2 in range(n_state):
            Tsas[s1, s2, s2] = T[s1, s2]
    return Tsas


def rewardvec_s_to_sas(R):
    """
    Reshapes a state reward vector (S) to a FiniteMDP matrix shape (SxAxS).
    NOTES: Unique action for every target state.
    """
    n_state = R.shape[0]
    Rsas = np.zeros((n_state, n_state, n_state))
    for s1 in range(n_state):
        for s2 in range(n_state):
            Rsas[s1, s2, s2] = R[s2]
    return Rsas


def smooth(x, window_len=10, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y




def generate_run_ID_PATFORM(options):
    '''
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in
    params.npy file.
    '''
    params = [
        options.MODEL_type,
        'res', str(options.res),
        'Ng', str(options.Ng),
        options.activation,
        'r', str(options.r),
        'lr', str(options.lr),
        'T', str(options.T),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID
