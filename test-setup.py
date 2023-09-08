
"""
Test Setup File.

This file should run just fine, with no problems.

You should see a little black box appear for a few seconds with a white dot moving around in it.
"""

import gymnasium as gym
import gym_moving_dot
import numpy as np
import torch

test_array = np.array([[1,2,3],[4,5,6]])
assert(test_array.shape == (2,3))

test_tensor = torch.tensor(test_array)
assert(test_tensor.shape == (2,3))

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("MovingDotDiscrete-v0", render_mode="human", step_size=2, random_start=True)
observation, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
