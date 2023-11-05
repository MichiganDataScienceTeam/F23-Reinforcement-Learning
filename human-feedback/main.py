"""
Human Feedback main.
"""
import os
import random
import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
from reinforceAgent import VanillaAgent
from rewardPredictor import RewardPredictor


def discount_rewards(rewards: list, gamma: float) -> torch.Tensor:
    """
    Calculate normalized discounted sum of rewards for episode.

    Args:
        rewards: array of rewards recieved during each timestep
        gamma: discount factor

    Returns:
        discounted_rewards: tensor of discounted sum of rewards for each timestep
    """
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0.0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted_rewards[t] = running_sum

    # discounted_rewards -= np.mean(discounted_rewards)
    # discounted_rewards /= np.std(discounted_rewards)
   
    return torch.from_numpy(discounted_rewards)


def generate_episode(env: gym.Env, policy: VanillaAgent, r_hat: RewardPredictor):
    """
    Generate episode in the given environment.

    Args:
        env: Gymnasium environment
        policy: policy valid to sample from in environment
    
    Returns:
        obs_action_list: list of (o, a, r) tuples
        log_actions: list of log action probs taking a in o
            torch.tensor
        r_hats: synthetic rewards for each (o,a) tuple
            python list

    """
    obs_a_data = []
    log_actions = []
    r_hats = []

    obs, info = env.reset()

    while True:
        # get action given obs
        action, action_log_probs = policy.select_action(obs)
        log_actions.append(action_log_probs)

        # get r_hat
        r = r_hat(obs, action)
        r_hats.append(r)

        # take action
        obs_new, reward, terminated, truncated, info = env.step(action)

        obs_a_data.append((np.array(obs_new), action, reward))

        obs = obs_new

        if terminated or truncated:
            break

    return obs_a_data, torch.tensor(log_actions), r_hats


def get_segments(episodes: list[tuple], number: int) -> list[tuple[list, list]]:
    """
    Get many pairs of segments from the list of episodes.

    Args:
        episodes: list of episodes (list of (o,a,r) tuples)
        number: number of selections to make

    Returns:
        segments: list of (sigma_1, sigma_2) tuples
    """
    segments = []
    n = len(episodes)
    assert(number < n and number > 0)

    seen_indices = set()

    while len(seen_indices) < number:
        indices = tuple(sorted(random.sample(range(n), 2)))
        if indices in seen_indices:
            continue
        
        seen_indices.add(indices)
        
        # make segments better
        sigma_1, sigma_2 = tuple([episodes[i] for i in indices])

        length_1, length_2 = len(sigma_1), len(sigma_2)
        if length_1 < length_2:
            sigma_2 = sigma_2[:length_1]
        elif length_1 > length_2:
            sigma_1 = sigma_1[:length_2]

        segments.append([sigma_1, sigma_2])
    
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
    gamma=0.99

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
    for i in tqdm(range(iterations)):
        # Process 1
        episodes = []
        log_actions = []
        r_hats = []
        num_episodes = 5
        for i in range(num_episodes):
            ep_data, ep_log_acts, ep_r_hats = generate_episode(env, agent)
            # generate_episode
            episodes.append(ep_data)
            log_actions.append(ep_log_acts)
            r_hats.append(ep_r_hats)
        # something to train the RL network
        r_hats = list(map(lambda x: discount_rewards(x, gamma), r_hats))

        # Process 2
        segments = get_segments(episodes, num_episodes - 1)
        mus = get_preferences(segments)

        # mu is a tuple, either (1,0), (0,1), or (0.5,0.5)
        dataset.extend([(*segments[i], mus[i]) for i in range(len(mus))])

        # Process 3
        # update r_hat on data
        loss = r_hat.calc_loss(dataset)
        r_hat.update(loss)

    