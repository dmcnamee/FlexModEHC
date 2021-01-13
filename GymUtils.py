#!/usr/bin/python

import numpy as np

def get_MDP_strucure(MDP):
    nS = MDP.env.nS
    nA = MDP.env.nA
    print('MDP has %i states and %i actions.'%(nS,nA))

    #reward and transition matrices
    T = np.zeros([nS, nA, nS])
    R = np.zeros([nS, nA, nS])
    for s in range(nS):
        MDP.env.state = s
        for a in range(nA):
            try:
                transitions = MDP.env.P[s][a]
                for p_trans,next_s,rew,done in transitions:
                    T[s,a,next_s] += p_trans
                    R[s,a,next_s] = rew
            except:
                next_s, rew, done, _ = MDP.env.step(a)
                T[s,a,next_s] += 1 # assumes deterministic transitions
                R[s,a,next_s] = rew
            T[s,a,:]/=np.sum(T[s,a,:])
    if np.any(np.isnan(T)):
        raise ValueError('Transitions cannot be NaNs.')
    return T,R,nS,nA

#transforms a deterministic policy in a policy matrix
def vectorize_policy(policy, nS, nA):
    new_policy = np.zeros([nS,nA])
    for s in range(nS):
        new_policy[s,policy[s]] = 1.0
    return new_policy

#calculate the mean reward received for each state under the current policy
def calculate_mean_reward(R, T, policy):
    if(len(policy.shape)==1):
        nS, nA, nS = T.shape
        policy = vectorize_policy(policy,nS,nA)

    return np.einsum('ijk,ijk,ij ->i', R, T, policy)

#calculate the transition probability under the given policy
def calculate_mean_transition(T, policy):
    if(len(policy.shape)==1):
        nS, nA, nS = T.shape
        policy = vectorize_policy(policy,nS,nA)

    return np.einsum('ijk,ij -> ik', T, policy)
