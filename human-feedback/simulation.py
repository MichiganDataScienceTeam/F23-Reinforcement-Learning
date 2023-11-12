import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from reinforceAgent import VanillaAgent


def main():
    # Set Torch Device
    device = "cpu"
    # Create Environment
    env = gym.make("LunarLander-v2", render_mode="human") # No render mode, for faster learning
    # env = gym.make("CartPole-v1")

    # Algorithm parameters
    num_episodes = 5

    # Agent parameters
    learning_rate = 0.001 # this is alpha... alpha > 0
    gamma = 0.97

    # Initialize Policy Parameter
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = VanillaAgent(n_features=n_features,
                        n_actions=n_actions,
                        learning_rate=learning_rate,
                        device=device)
    
    # Load Weights
    path = "weights/first_weights.h1"
    agent.agent.load_state_dict(torch.load(path))
    agent.agent.eval()

    # Run Simulation
    for _ in range(num_episodes):
        # get an initial state
        state, _ = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                action, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, _, terminated, truncated, _ = env.step(action)

            # update if the environment is done
            done = terminated or truncated
    
    env.close()


if __name__=="__main__":
    main()