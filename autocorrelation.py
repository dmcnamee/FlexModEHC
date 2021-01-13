#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from numpy.matlib import repmat
from scipy.stats import sem
from scipy.optimize import LinearConstraint, Bounds

tol = 10**-3 # tolerance for bounds and linear constraints


def zcf_gen(s, W, deltaT, rho):
    """ FUNCTION: computes generated autocorrelation
    INPUTS: s       = spectrum
            W       = spectral components (n_k,n_s,n_s)
            deltaT  = lags to compute
            rho     = state probability vector
    """
    deltaT = np.asarray(deltaT)
    n_s = rho.size
    n_t = deltaT.size
    Wd = np.array([W[:,i,i] for i in range(n_s)]).T # (n_s,n_k)
    ZCgen = repmat(s.reshape(-1,1), 1, n_t) # (n_k,n_t) convention
    ZCgen = ZCgen**deltaT
    ZCgen = Wd@ZCgen # (n_s,n_k) x (n_k,n_t) -> (n_s,n_t)
    ZCgen = rho@ZCgen # (,n_s) x (n_s,n_t) -> (n_t,)
    return ZCgen


def zcf_sum(s, W, deltaT, rho):
    """ sums over zcf components """
    ZCgen = zcf_gen(s, W, deltaT, rho)
    return ZCgen.sum()


def acf_gen(s, W, T, deltaT, rho):
    """ FUNCTION: computes generated autocorrelation
    INPUTS: s       = spectrum
            W       = spectral components (n_k,n_s,n_s)
            T       = max time to integrate over
            deltaT  = lags to compute
            rho     = state probability vector
    """
    deltaT = np.asarray(deltaT)
    n_s = rho.size
    n_t = deltaT.size
    Wd = np.array([W[:,i,i] for i in range(n_s)]).T # (n_s,n_k)
    ACgen = np.zeros((T+1,n_t))
    for t in range(T+1):
        rho_t = np.einsum('i, kij, k', rho, W, s**t); rho_t = rho_t/rho_t.sum()
        ACgen_t = repmat(s.reshape(-1,1), 1, n_t) # (n_k,n_t) convention
        ACgen_t = ACgen_t**deltaT
        ACgen_t = Wd@ACgen_t # (n_s,n_k) x (n_k,n_t) -> (n_s,n_t)
        ACgen_t = rho_t@ACgen_t # (,n_s) x (n_s,n_t) -> (n_t,)
        ACgen[t,:] = ACgen_t
    return ACgen

def acf_sum(s, W, T, deltaT, rho, sumT=True, sum_deltaT=True):
    """ sums over acf components """
    ACgen = acf_gen(s, W, T, deltaT, rho)
    if sumT:
        if sum_deltaT:
            return ACgen.sum()
        else:
            return ACgen.sum(0)
    else:
        if sum_deltaT:
            return ACgen.sum(1)
        else:
            return ACgen


def constraints_stochmat(W, tol=tol):
    """
    FUNCTION: linear constraints which ensure resulting evolution matrix is a stochastic matrix
    INPUTS: W = (n_k, n_s, n_s) spectral weights
    """
    n_k = W.shape[0]
    # sum_{j,k} W[k,i,j]*s_k = 1
    lc1_stochmat = LinearConstraint(A=W.T.sum(0), lb=1-tol, ub=1+tol)
    # sum_{k} W[k,i,j]*s_k >= 0
    lc2_stochmat = LinearConstraint(A=W.T.reshape((-1, n_k)), lb=-tol, ub=np.inf)
    return (lc1_stochmat, lc2_stochmat)

# constraints as functions
def constraint_func1(x):
    """ equal 1 """
    return (W.T.sum(0).dot(x)==1).all()

def constraint_func2(x):
    """ greater than -1 """
    return (W.T.reshape((-1, n_k)).dot(x) > -1-tol).all()

def constraint_func3(x):
    """ less than 1 """
    return (W.T.reshape((-1, n_k)).dot(x) < 1 + tol).all()


def bounds_statdist(n_k, tol=tol):
    """
    FUNCTION: bounds to ensure convergence to stationary state distribution
    INPUT: n_k = number of e-vectors
    """
    # s_1 = 1
    # |s_k| <= 1
    lb = np.ones((n_k,))*(-1-tol)
    ub = np.ones((n_k,))*(1+tol)
    lb[0] = 1
    ub[0] = 1
    return Bounds(lb=lb, ub=ub)


def roll_pad(x, n, fill_value=0):
    """
    FUNCTION: rolls matrix x along columns
    """
    if n == 0:
        return x
    elif n < 0:
        n = -n
        return np.fliplr(np.pad(np.fliplr(x),((0,0),(n,0)), mode='constant', constant_values=fill_value)[:, :-n])
    else:
        return np.pad(x,((0,0),(n,0)), mode='constant', constant_values=fill_value)[:, :-n]


def estimate_occ_acf(data, d=0):
    """
    FUNCTION: estimate occupator autocorrelation
    INPUTS: data = (n_t, n_samp) matrix of samples
    """
    n_t = data.shape[0]
    n_samp = data.shape[1]
    n_state = data.max()+1

    AC_samp = np.zeros((n_t,n_samp))
    for k in range(n_t):
        X = roll_pad(data.T, -k, (d+1)*(n_state+1)).T
        N = float(n_t-k)
        if d == 0:
            AC_samp[k,:] = (X==data).sum(0)/N
        else:
            AC_samp[k,:] = (np.abs(X-data)<=d).sum(0)/N
    AC = AC_samp.mean(1)
    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)
    # use variance across samples as a sample variance (does not take into account within-sample, cross-lag variance)
    return AC, AC_sem


def estimate_occ_zero_cf(data, d=0):
    """
    INPUTS: estimates occupator correlation with zero-time occupator.
    INPUTS: data = (n_t, n_samp) matrix of samples
    """
    n_t = data.shape[0]
    n_samp = data.shape[1]

    AC_samp = np.zeros((n_t,n_samp))
    for k in range(n_t):
        if d == 0:
            AC_samp[k,:] = (data[0,:]==data[k,:])
        else:
            AC_samp[k,:] = (np.abs(data[0,:]-data[k,:])<=d)
    AC = AC_samp.mean(1)
    if n_samp > 1:
        AC_sem = sem(AC_samp, axis=1)
    else:
        AC_sem = np.zeros(AC.shape)
    # use variance across samples as a sample variance (does not take into account within-sample, cross-lag variance)
    return AC, AC_sem
