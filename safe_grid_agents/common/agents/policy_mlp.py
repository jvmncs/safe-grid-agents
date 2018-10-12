"""Policy gradient and actor-critic agents."""
from .policy_base import PPOBaseAgent

from typing import Tuple
import torch
import torch.nn as nn


class PPOMLPAgent(PPOBaseAgent):
    """Actor-critic variant of PPO."""

    def __init__(self, env, args) -> None:
        self.n_layers = args.n_layers
        self.n_hidden = args.n_hidden
        super().__init__(env, args)

    def build_ac(self) -> None:
        """Build the fused actor-critic architecture."""
        first = nn.Sequential(nn.Linear(self.n_input, self.n_hidden), nn.ReLU())
        hidden = nn.Sequential(
            *tuple(
                nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU())
                for _ in range(self.n_layers - 1)
            )
        )
        self.network = nn.Sequential(first, hidden)
        self.actor = nn.Linear(self.n_hidden, int(self.action_n))
        self.critic = nn.Linear(self.n_hidden, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(
            x, requires_grad=False, dtype=torch.float32, device=self.device
        )
        if len(x.shape) <= 2:
            x = x.flatten().reshape(1, -1)
        else:
            # flatten everything excapt for batch dimension
            x = x.reshape(x.shape[0], -1)

        x = self.network(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
