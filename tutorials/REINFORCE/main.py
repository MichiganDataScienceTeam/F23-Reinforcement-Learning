"""REINFORCE Policy-Gradient for Lunar Landing"""
import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from reinforceAgent import VanillaAgent


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

    # Agent parameters
    learning_rate = 0.001 # this is alpha... alpha > 0
    gamma = 0.97  # oddly, HAS to be less than 1

    # Initialize Policy Parameter
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = VanillaAgent(n_features=n_features,
                         n_actions=n_actions,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         device=device)

    # Training Loop
    for _ in tqdm(range(num_episodes)):
        # Generate Episode
        # TODO

        # Learn from episode, all at once because gradient changes at each step
        # Calculate G's for each state efficiently
        # TODO

        # Calculate Policy Loss
        # TODO

        # Add graphing data
        # TODO
        pass

    plot_training_data(total_rewards, losses)
    # Save network
    # save_network(agent)
    return


def generate_episode(env: gym.Env, agent: VanillaAgent) -> tuple[torch.Tensor, list]:
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
    while True:
        # Decide and Save action
        # TODO

        # Take action and save reward
        # TODO
        terminated = True  # remove this when you're coding
        truncated = True  # remove this when you're coding

        if terminated or truncated:
            break

    return torch.stack(log_actions), rewards


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

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
   
    return torch.from_numpy(discounted_rewards)


def save_network(agent: VanillaAgent):
    """Save network parameters and network weights."""
    if not os.path.exists("weights"):
        os.mkdir("weights")

    torch.save(agent.agent.state_dict(), "weights/reinforce_weights.h5")


def plot_training_data(rewards: list[float], losses: list[torch.Tensor]):
    """Plot the training data."""
    if not os.path.exists("figures"):
        os.mkdir("figures")
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
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
    axs[1].set_xlabel("Number of episodes")
        
    plt.savefig("figures/training_loss.png")


if __name__=="__main__":
    main()
