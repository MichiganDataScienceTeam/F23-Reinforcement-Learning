"""
Human Feedback main.
"""
import os
import random
import gymnasium as gym
import torch
import numpy as np
from reinforceAgent import VanillaAgent
from rewardPredictor import RewardPredictor

def generate_episode(env: gym.Env, policy: VanillaAgent):
    """
    Generate episode in the given environment.

    Args:
        env: Gymnasium environment
        policy: policy valid to sample from in environment
    
    Returns:
        obs_action_list: list of (o, a, r) tuples

    """
    obs_a_data = []

    obs, info = env.reset()

    while True:
        # get action given obs
        # TODO, think about log_probs and if we need them here
        action, action_log_probs = policy.select_action(obs)

        # take action
        obs_new, reward, terminated, truncated, info = env.step(action)

        obs_a_data.append((obs, action, reward))

        obs = obs_new

        if terminated or truncated:
            break

    return obs_a_data


def get_segments(episodes: list[tuple], number: int) -> list[tuple[list, list]]:
    """
    Get many pairs of segments from the list of episodes.

    Args:
        episodes: list of episodes (list of (o,a,r) tuples)
        number: number of selections to make

    Returns:
        segments: list of (simga_1, sigma_2) tuples
    """
    segments = []
    n = len(episodes)
    assert(number < n and number > 0)

    for _ in range(number):
        indices = sorted(random.sample(range(n), 2))
        # make segments better
        sigmas = tuple([episodes[i] for i in range(indices)])
        segments.append(sigmas)
    
    return segments


def get_preferences(segments: list[tuple[list,list]]):
    """
    Use synthetic oracle to pick preferred segment per tuple.

    Args:
        segments: list of tuples. Each tuple is a pair of episode segments of the same length.
    
    Returns:
        mus: list of preferences for each segment pair
    """
    mus = []
    for sigma_1, sigma_2 in segments:
        mu = (0.5,0.5)
        # forward each o,a in each sigma to r_hat
        sigma_1_true_sum = np.sum([i[2] for i in sigma_1])
        sigma_2_true_sum = np.sum([i[2] for i in sigma_2])

        if sigma_1_true_sum > sigma_2_true_sum:
            mu = (1,0)
        elif sigma_2_true_sum > sigma_1_true_sum:
            mu = (0,1)
        
        mus.append(mu)
    
    assert(len(segments) == len(mus))

    return mus

if __name__=="__main__":
    """Main."""
    # Environment
    env = gym.make("LunarLander-v2")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Neural Network Parameters
    device = "cpu"
    agent_lr = 1e-3
    r_hat_lr = 1e-3

    agent = VanillaAgent(n_features=n_observations,
                         n_actions=n_actions,
                         learning_rate=agent_lr,
                         device=device) # Fill with agent

    r_hat = RewardPredictor(n_features=n_observations,
                            n_actions=n_actions,
                            learning_rate=r_hat_lr,
                            device=device) # Fill with r_hat

    dataset = []

    iterations = 50
    for i in range(iterations):
        # Process 1
        episodes = []
        num_episodes = 5
        for i in range(num_episodes):
            ep_data = generate_episode(env, agent) # generate_episode
            episodes.append(ep_data)
        
        # Process 2
        segments = get_segments(episodes)
        mus = get_preferences(segments)
        # mu is a tuple, either (1,0), (0,1), or (0.5,0.5)
        dataset.extend([(segments[i][0], segments[i][1], mus[i]) for i in range(len(mus))])

        # Process 3
        # update r_hat on data
        loss = r_hat.calc_loss(dataset)
        r_hat.update(loss)
