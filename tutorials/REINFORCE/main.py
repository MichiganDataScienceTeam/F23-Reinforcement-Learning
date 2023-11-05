"""REINFORCE Policy-Gradient for Lunar Landing"""
import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from reinforceAgent import VanillaAgent
from rewardPredictor import RewardPredictor

import math
import random


def main() -> None:
    """Main for Vanilla Policy-Gradient."""
    # Set Torch Device
    device = "cpu"
    # Create Environment
    env = gym.make("LunarLander-v2") # No render mode, for faster learning
    # env = gym.make("CartPole-v1")

    # Algorithm parameters
    num_episodes = 1000
    total_rewards = []
    losses = []
    predictor_losses = []

    # Agent parameters
    learning_rate = 0.001 # this is alpha... alpha > 0
    gamma = 0.99  # oddly, HAS to be less than 1

    # Initialize Policy Parameter
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = VanillaAgent(n_features=n_features,
                         n_actions=n_actions,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         device=device)
    
    reward_predictor = RewardPredictor(
        n_features=8,
        n_actions=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        device=device
    )


    # Training Loop
    for counter in tqdm(range(num_episodes)):
        # Generate Episode
        # TODO
        log_actions_1, true_rewards_1, r_hat_rewards_1, p1 = generate_episode(env, agent, reward_predictor)
        log_actions_2, true_rewards_2, r_hat_rewards_2, p2 = generate_episode(env, agent, reward_predictor)

        
        #get minimum length of the two episodes and use that as trajectory length
        min_len = min(len(r_hat_rewards_1), len(r_hat_rewards_2))
        #if (counter  % 10 == 0):
            #print(min_len)
        

        """
        cap = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        r_hat_rewards_1 = r_hat_rewards_1[math.floor(min_len * (cap - 0.1)):math.floor(min_len * cap)]
        r_hat_rewards_2 = r_hat_rewards_2[math.floor(min_len * (cap - 0.1)):math.floor(min_len * cap)]
        log_actions_2 = log_actions_2[math.floor(min_len * (cap - 0.1)):math.floor(min_len * cap)]
        true_rewards_1 = true_rewards_1[math.floor(min_len * (cap - 0.1)):math.floor(min_len * cap)]
        true_rewards_2 = true_rewards_2[math.floor(min_len * (cap - 0.1)):math.floor(min_len * cap)]
        """

        r_hat_rewards_1 = r_hat_rewards_1[:min_len]
        r_hat_rewards_2 = r_hat_rewards_2[:min_len]
        log_actions_1 = log_actions_1[:min_len]
        log_actions_2 = log_actions_2[:min_len]
        true_rewards_1 = true_rewards_1[:min_len]
        true_rewards_2 = true_rewards_2[:min_len]

        true_sum_1 = discount_rewards(true_rewards_1, gamma)[0]
        true_sum_2 = discount_rewards(true_rewards_2, gamma)[0]


        oracle_choice = 1 if true_sum_2 > true_sum_1 else (0.5 if true_sum_2 == true_sum_1 else 0)
        mu2 = oracle_choice
        mu1 = 1 - oracle_choice


        # Learn from episode, all at once because gradient changes at each step
        # Calculate G's for each state efficiently
        # TODO
        r_hat_modified = [element.item() for element in r_hat_rewards_1]
        gs = discount_rewards(r_hat_modified, gamma, normalize=True)
        #print(gs.size())
        #print(log_actions.size())
        loss = agent.calc_loss(gs, log_actions_1)
        #print(loss.item())
        


        #r_hat_modified_2 = [element2.item() for element2 in r_hat_rewards_2]
        #gamma_2 = gamma
        #gs_2 = discount_rewards(r_hat_modified_2, gamma_2, normalize=True)
        #print(gs.size())
        #print(log_actions.size())
        #loss_2 = agent.calc_loss(gs_2, log_actions_2)
        #print("AGENT LOSSES", loss, loss_2)
        #print(loss.item())
        #agent.update(loss_2)
        agent.update(loss)


        # Calculate Policy Loss
        # TODO
        policy_loss = loss

        #Update reward predictor
        predictor_loss = reward_predictor.calc_loss(mu1, mu2, 
                                                    r_hat_rewards_1,
                                                    r_hat_rewards_2,
                                                    )
        reward_predictor.update(predictor_loss)

        # Add graphing data
        # TODO
        total_rewards.append(true_rewards_1[0])
        losses.append(policy_loss)
        predictor_losses.append(predictor_loss)

    plot_training_data(total_rewards, losses, predictor_losses)
    # Save network
    save_network(agent)
    return


def generate_episode(env: gym.Env, agent: VanillaAgent, reward_predictor: RewardPredictor) -> tuple[torch.Tensor, list]:
    """
    Generate episode following policy without updating policy.
    Doesn't matter if you assume that rewards are recieved in same timestep as action is taken or not.
    e.g. s_0, a_0, r_0, s_1, a_1... same as
         s_0, a_0, r_1, s_1, a_1 ...

    Args:
        env: gymnasium LunarLander Envrionment
        agent: Vanillaagent agent class

    Returns:
        log_actions: tensor of all (ln pi(A_t | S_t))
        rewards: list of R_t recieved for each timestep
    """
    obs, _ = env.reset()
    log_actions = []
    rewards = []
    r_hat_rewards = []
    all_actions = []
    while True:
        # Decide and Save action
        # TODO
        action, log_probs = agent.select_action(obs)
        all_actions.append(action)


        # Take action and save reward
        # TODO
        state, reward, terminated, truncated, _ = env.step(action.item())
        input_array = state.flatten()
        np.append(input_array, action.item())
        predicted_reward = reward_predictor.get_reward(input_array)
        r_hat_rewards.append(predicted_reward)

        #print("R:")
        #print(reward)
        rewards.append(reward)
        log_actions.append(log_probs)
        #terminated = False  # remove this when you're coding
        #truncated = False  # remove this when you're coding

        if terminated or truncated:
            break

    
    return torch.stack(log_actions), rewards, r_hat_rewards, all_actions


def discount_rewards(rewards: list, gamma: float, normalize=False) -> torch.Tensor:
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

    if normalize:
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
   
    return torch.from_numpy(discounted_rewards)


def save_network(agent: VanillaAgent):
    """Save network parameters and network weights."""
    if not os.path.exists("weights"):
        os.mkdir("weights")

    torch.save(agent.agent.state_dict(), "weights/reinforce_weights.h5")


def plot_training_data(rewards: list[float], losses: list[torch.Tensor], predictor_losses: list[torch.Tensor]):
    """Plot the training data."""
    if not os.path.exists("figures"):
        os.mkdir("figures")
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    fig.suptitle("Training Plots for REINFORCE agent in LunarLander-v2 Environment.")
    # Rewards
    axs[0].set_title("Episodic Cumulative Returns")
    axs[0].plot(
        np.arange(len(rewards)),
        rewards,
    )
    axs[0].set_xlabel("Number of episodes")
    # Loss
    axs[1].set_title("Agent Loss")
    axs[1].plot(
        np.arange(len(losses)),
        [l.detach().numpy() for l in losses],
    )

    axs[2].set_xlabel("Number of episodes")

    axs[2].set_title("Predictor Loss")
    axs[2].plot(
        np.arange(len(predictor_losses)),
        [l.detach().numpy() for l in predictor_losses],
    )
    axs[2].set_xlabel("Number of episodes")
        
    plt.savefig("figures/training_loss.png")


if __name__=="__main__":
    main()
