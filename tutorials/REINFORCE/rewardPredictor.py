"""Vanilla Policy-Gradient Agent Torch Framework."""
import os
import torch
import torch.nn as nn
import numpy as np
import math

torch.autograd.set_detect_anomaly(True)

class RewardPredictor(nn.Module):
    """
    REINFORCE agent Class.

    Args:
        n_features: number of features of input state.
        n_actions: number of actions in environment agent can take.
        agent_lr: learning rate for agent policy network
        alpha: step_size of agents policy
        gamma: discount factor for environment
        device: device to run computations on
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        device: torch.device,
    ) -> None:
        """Initializes the model architecture and policy parameters."""
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.log_probs = []

        layers = [
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ]

        self.agent = nn.Sequential(*layers).to(self.device)
        self.agent.requires_grad_(True)

        self.agent_optim = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Forward pass of agent network.

        Args: 
            x: a vector of states

        Returns:
            action_logits: A tensor with action logits, with shape [1, n_actions].
        """
        # TODO
        action_logits = self.agent(x)
        return action_logits

    def update(self, loss: torch.Tensor) -> None:
        """
        Backwawrd pass of agent network.

        Args: 
            loss: a tensor of agents' loss with shape [1,]
        """
        # TODO
        self.agent_optim.zero_grad()
        loss.backward(retain_graph = True)
        #print(loss.item())
        self.agent_optim.step()

    def get_reward(self, state_action_list):
        """
        Action selection for agent

        Args:
            x: a vector of states with shape[1,n_features]

        Returns:
            actions: a tensor representing the selected actions with shape [1, 1]
            action_log_probs: log probability of selected actions
        """
        # TODO
        out = self.forward(torch.from_numpy(state_action_list))
        return out

    def calc_loss(self, mu1: float, mu2: float, r_hat_1: list, r_hat_2: list) -> torch.Tensor:
        """
        Calculate loss

        Args:
            gs: discounted sum of rewards for each timestep
            log_probs: action log probs at each timestep
        """

        # TODO
        predicted_probability_1 = torch.exp(sum(r_hat_1)) / (torch.exp(sum(r_hat_1)) + torch.exp(sum(r_hat_2)))
        predicted_probability_2 = torch.exp(sum(r_hat_2)) / (torch.exp(sum(r_hat_1)) + torch.exp(sum(r_hat_2)))
        #print(predicted_probability_1, predicted_probability_2)
        loss = -1 * (mu1 * torch.log(predicted_probability_1) + mu2 * torch.log(predicted_probability_2))
        #print("MU'S: ", mu1, mu2)
        #print("LOSS ", loss)

        return loss
