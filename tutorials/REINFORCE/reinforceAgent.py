"""Vanilla Policy-Gradient Agent Torch Framework."""
import os
import torch
import torch.nn as nn
import numpy as np


class VanillaAgent(nn.Module):
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
            # TODO
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
        action_logits = torch.Tensor(0)
        return action_logits

    def update(self, loss: torch.Tensor) -> None:
        """
        Backwawrd pass of agent network.

        Args: 
            loss: a tensor of agents' loss with shape [1,]
        """
        # TODO
        return

    def select_action(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Action selection for agent

        Args:
            x: a vector of states with shape[1,n_features]

        Returns:
            actions: a tensor representing the selected actions with shape [1, 1]
            action_log_probs: log probability of selected actions
        """
        # TODO
        action = 0
        action_log_probs = 0

        return (action, action_log_probs)

    def calc_loss(self, gs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss

        Args:
            gs: discounted sum of rewards for each timestep
            log_probs: action log probs at each timestep
        """
        assert(gs.size() == log_probs.size())

        # TODO
        loss = 0.0
        
        return loss
