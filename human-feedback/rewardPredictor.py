"""Vanilla Policy-Gradient Agent Torch Framework."""
import os
import torch
import torch.nn as nn
import numpy as np


class RewardPredictor(nn.Module):
    """
    Reward Predictor Neural Network Module.

    Args:
        n_features: number of features of input state.
        n_actions: number of actions in environment agent can take.
        learning_rate: learning rate for reward network
        alpha: step_size of agents policy
        device: device to run computations on
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        """Initializes the model architecture and policy parameters."""
        super().__init__()
        self.device = device

        self.n_features = n_features
        self.n_actions = n_actions

        layers = [
            nn.Linear(n_features+n_actions, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        ]

        self.network = nn.Sequential(*layers).to(self.device)
        self.network.requires_grad_(True)

        self.agent_optim = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, obs: np.ndarray, action: np.ndarray) -> torch.Tensor:
        """
        Forward pass of reward network.

        Args: 
            obs: observation from environment
            action: action taken in response to the observation

        Returns:
            predicted_reward: predicted reward.
        """
        one_hot_action = np.zeros((self.n_actions))
        one_hot_action[action] = 1
        inputs = np.concatenate((obs, one_hot_action))
        x = torch.Tensor(inputs)
        predicted_reward = self.network(x)
        return predicted_reward

    def update(self, loss: torch.Tensor) -> None:
        """
        Backward pass of agent network.

        Args: 
            loss: a tensor of agents' loss with shape [1,]
        """
        self.agent_optim.zero_grad()
        loss.backward()
        self.agent_optim.step()
        return

    def calc_loss(self, dataset) -> torch.Tensor:
        """
        Calculate cross entropy loss.

        Args:
            x
        """
        targets = torch.tensor([item[2] for item in dataset], dtype=torch.float)
        seg_1_sums = torch.stack([sum([self.forward(o,a) for (o,a,_) in item[0]]) for item in dataset])
        seg_2_sums = torch.stack([sum([self.forward(o,a) for (o,a,_) in item[1]]) for item in dataset])
        logits = torch.cat([seg_1_sums,seg_2_sums], dim=1)
        return self.loss(logits, targets)
