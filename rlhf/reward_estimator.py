"""Estimate the reward of a trajectory using a neural network fit to the discriminator."""
"""This computes r-hat."""

import math
import gymnasium
import torch
import torch.nn as nn
from discriminator import compare
from typing import Tuple, List, TypeVar

from _types import Trajectory

#class reward(nn.module)
#Create a function that initializes the parameters of the neural network
#
class Rewards(nn.Module):
    def __init__(
                self,
                environment: gymnasium.Env,
                learning_rate: float,
                device: torch.device,
                ) -> None:
        super().__init__()
        self.device = device
        assert environment.observation_space.shape is not None, "Missing observation space shape"
        # NOTE: This assumes that the action space is Discrete(4), as with LunarLander.
        self.layers = [
            nn.Linear(environment.observation_space.shape[0] + 4, 64),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        ]
        self.agent = nn.Sequential(*self.layers).to(self.device)
        self.agent.requires_grad_(True)
        self.agent_optim = torch.optim.Adam(self.agent.parameters(), lr = learning_rate)

    def r_hat(self, X: torch.Tensor):
        return self.agent(X.to(self.device))
    
    # build X over all time steps
    def build_X(self, t: Trajectory) -> torch.Tensor:
        lines = []
        for obs, act, _ in t:
            act_onehot = [0] * 4
            act_onehot[act] = 1
            line = torch.concat([obs.flatten(), torch.Tensor(act_onehot)])
            lines.append(line)
        return torch.stack(lines)
    
    def estimate_reward(self, trajectory: Trajectory) -> torch.Tensor:
        return self.agent(self.build_X(trajectory).to(self.device))

    def P_hat(self, t1: Trajectory, t2: Trajectory) -> torch.Tensor:
        t1_r_exp = torch.exp(torch.sum(self.r_hat(self.build_X(t1))))
        t2_r_exp = torch.exp(torch.sum(self.r_hat(self.build_X(t2))))

        return (t1_r_exp / (t1_r_exp + t2_r_exp))
        
    def loss(self, trajectory_pairs: List[Tuple[Trajectory, Trajectory, Tuple[float, float]]]) -> torch.Tensor:
        # trajectory_pairs represents the database of human feedback
        # Each tuple (trajectory, trajectory, mu) represents a pair with feedback
        loss = torch.Tensor(0)
        for element in trajectory_pairs:
            loss += element[2][0] * torch.log(self.P_hat(element[0], element[1])) + element[2][1] * torch.log(self.P_hat(element[1], element[0]))
        loss *= -1
        return loss
    
    def update(self, loss: torch.Tensor):
        self.agent_optim.zero_grad()
        loss.backward()
        self.agent_optim.step()

