"""Solve the gridworld using Monte Carlo."""

import random
import numpy as np
from gridworld import GridWorld

def init_ep_soft_policy(epsilon: float):
    """
    Return random policy.
    
    Policy is a dictionary of:
        key: state
        val: list of probs of taking each action
    
    Could just as easily make policy a 16x4 numpy array.
    Not strict here because we include unreachable states in the policy
    """
    policy = {}
    for state in range(16):
        probs = [epsilon/4 for _ in range(4)]
        probs[ np.random.randint(4) ] += 1-epsilon
        policy[state] = probs
    return policy

def sample_policy(policy, state: int):
    '''
    list_probs: list of probabilites that add up to 1

    Returns: index corresponding to which bin a random number fell into
    '''
    list_probs = policy[state]
    length = len(list_probs)
    # cummulative array of probabilites
    cu_list = [sum(list_probs[0:x:1]) for x in range(0, length+1)][1:]

    choice = random.random()
    for i in range(length):
        if choice <= cu_list[i]:
            return i

    print('unsafe pick_action')
    return -1

def generate_episode(policy, random_start=True) -> list[tuple[int, int, int]]:
    """
    Returns: list of (state, action, reward) tuples
    """
    episode = []
    env = GridWorld(random_start)
    obs = env.reset()

    while True:
        action = sample_policy(policy, obs)
        new_obs, reward, truncated, terminated = env.step(action)

        episode.append( (obs, action, reward) )
        obs = new_obs

        if truncated or terminated:
            break
    return episode

def init_q_func() -> np.ndarray:
    """
    Instead of a dict, use numpy array
    """
    return np.full(shape=(16,4), fill_value=0)

def get_return(partial_episode: list[tuple[int, int, int]], gamma:float) -> float:
    '''
    partial_episode: list of (s_i, a_i, r_i+1) tuples
    gamma: discount factor, between 0 and 1
    '''
    returns = 0
    for i in range(len(partial_episode)):
        _,_,this_return = partial_episode[i]
        returns += pow(gamma,i)*this_return
    return returns


if __name__=="__main__":
    # Hyperparameters
    max_iterations = 500
    iteration = 0
    gamma = 0.97
    epsilon = 0.05
    # Initialize data structures
    returns = {(s, a):[] for s in range(16) for a in range(4)}
    policy = init_ep_soft_policy(epsilon)
    q_func = init_q_func()

    print("Policy Iteration: Monte Carlo")

    while iteration < max_iterations:
        iteration += 1
        
        # 1: Generate Episode
        # TODO

        # 2: Extract information from episode
        # TODO

        # 3: Update policy
        # TODO

    print(q_func)
    print(policy)
