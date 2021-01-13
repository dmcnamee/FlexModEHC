#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from GymUtils import *
from utils import row_norm

# https://gist.github.com/lucisdp/1b07c63dda740d50f91e6e44403b7072
def policy_evaluation(R, T, policy, max_eval=1000, gamma=1.0):
    """
    Evaluates value function of current policy using Bellman iterations.
    Synchronously sweeps entire state-space in a vectorized fashion.
    INPUTS:
    R           rewards for every transition
    T           transition probabilities
    policy      initial policy
    max_eval    maximum number of evaluation sweeps
    gamma       discount factor (< 1 for convergence guarantees)
    OUTPUTS:
    V           value function
    """

    #calculate mean reward and the mean transition matrix
    mean_R = calculate_mean_reward(R, T, policy)
    mean_T = calculate_mean_transition(T, policy)

    #initializes value function to 0
    V = np.zeros(mean_R.shape)

    #iterate k times the Bellman Equation
    for i in range(max_eval):
        V = mean_R + gamma * np.dot(mean_T, V)

    return V


# https://gist.github.com/lucisdp/1b07c63dda740d50f91e6e44403b7072
def policy_iteration(R, T, policy=None, max_iter=100, max_eval=100, gamma=0.99):
    """
    Iteratively improves policy by applying max operation to value function.
    Synchronously sweeps entire state-space in a vectorized fashion.
    INPUTS:
    R           rewards for every transition
    T           transition probabilities
    policy      initial policy
    max_iter    maximum number of iterations
    max_eval    maximum number of evaluation sweeps
    gamma       discount factor (< 1 for convergence guarantees)
    OUTPUTS:
    policy      optimized policy
    """

    nS,nA,nS = T.shape

    if policy is None:
        policy = np.ones((nS,nA))
        policy = row_norm(policy)
        # policy = np.ones((nS,)).astype('int')

    for _ in range(max_iter):
        #store current policy
        opt = policy.copy()

        #evaluate value function (at least approximately)
        V = policy_evaluation(R, T, policy, max_eval, gamma)

        #calculate Q-function
        Q = np.einsum('ijk,ijk->ij', T, R + gamma * V[None,None,:])

        #update policy
        policy = np.argmax(Q, axis=1)

        #if policy did not change, stop
        if np.array_equal(policy,opt):
            break

    return vectorize_policy(policy,nS,nA)
