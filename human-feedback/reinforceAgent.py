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
        self.log_probs = []

        layers = [
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_actions)
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
        x = torch.Tensor(x)
        action_logits = self.agent(x)
        return action_logits

    def update(self, loss: torch.Tensor) -> None:
        """
        Backwawrd pass of agent network.

        Args: 
            loss: a tensor of agents' loss with shape [1,]
        """
        self.agent_optim.zero_grad()
        loss.backward()
        self.agent_optim.step()
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
        action_logits = self.forward(x)
        action_pdf = torch.distributions.Categorical(logits=action_logits)
        action = action_pdf.sample()
        action_log_probs = action_pdf.log_prob(action)

        return (action.item(), action_log_probs)

    def calc_loss(self, gs: list[torch.Tensor], log_probs: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss. The 0th dimension matches. The length of each episode may be different.
        So we pad the 1th dimension with zeroes.

        Args:
            gs: discounted sum of rewards for each timestep
                dimensionality is (# eps, ep length)
            log_probs: action log probs at each timestep
        """
        # assert(gs.size() == log_probs.size())
        
        pad_gs = torch.nn.utils.rnn.pad_sequence(gs)
        pad_probs = torch.nn.utils.rnn.pad_sequence(log_probs)

        loss = -torch.sum(pad_gs * pad_probs)
        
        return loss
