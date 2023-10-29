"""Estimate the reward of a trajectory using a neural network fit to the discriminator."""
"""This computes r-hat."""

import math
import gymnasium
import torch
import torch.nn as nn
from discriminator import compare, ObsType, ActType
from typing import Tuple, List, TypeVar


Trajectory = List[Tuple[ObsType, ActType, float]]

#class reward(nn.module)
#Create a function that initializes the parameters of the neural network
#
class Rewards(nn.Module):
    def __init__(
                self,
                environment: gymnasium.Env,
                learning_rate: float,
                gamma: float,
                device: torch.device,
                ) -> None:
        super().__init__()
        self.gamma = gamma
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

    def P_hat(self, t1: Trajectory, t2: Trajectory) -> float:
        # build X over all time steps
        def build_X(t: Trajectory) -> torch.Tensor:
            lines = []
            for obs, act, _ in t:
                act_onehot = [0] * 4
                act_onehot[act] = 1
                line = torch.concat([obs.flatten(), torch.Tensor(act_onehot)])
                lines.append(line)
            return torch.stack(lines)
        
        t1_r_exp = torch.exp(torch.sum(self.r_hat(build_X(t1))))
        t2_r_exp = torch.exp(torch.sum(self.r_hat(build_X(t2))))

        return (t1_r_exp / (t1_r_exp + t2_r_exp)).item()
        
    def loss(self, trajectory_pairs: List[Tuple[Trajectory, Trajectory, Tuple[float, float]]]):
        # trajectory_pairs represents the database of human feedback
        # Each tuple (trajectory, trajectory, mu) represents a pair with feedback
        loss = 0
        for element in trajectory_pairs:
            loss += element[2][0] * math.log(self.P_hat(element[0], element[1])) + element[2][1] * math.log(self.P_hat(element[1], element[0]))
        loss *= -1
        return loss
        


    
