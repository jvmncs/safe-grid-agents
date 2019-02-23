"""Policy gradient and actor-critic agents."""
from safe_grid_agents.common.agents.policy_base import PPOBaseAgent

from typing import Tuple
import torch
import torch.nn as nn


class PPOCNNAgent(PPOBaseAgent):
    """Actor-critic variant of PPO."""

    def __init__(self, env, args) -> None:
        self.n_channels = args.n_channels
        self.n_layers = args.n_layers
        super().__init__(env, args)

    def build_ac(self) -> None:
        """Build the fused actor-critic architecture."""
        in_channels = self.board_shape[0]
        first = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, self.n_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        hidden = nn.Sequential(
            *tuple(
                nn.Sequential(
                    torch.nn.Conv2d(
                        self.n_channels,
                        self.n_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
                for _ in range(self.n_layers - 1)
            )
        )
        self.network = nn.Sequential(first, hidden)
        self.bottleneck = nn.Conv2d(
            in_channels, self.n_channels, kernel_size=1, stride=1
        )

        self.actor_cnn = nn.Sequential(
            torch.nn.Conv2d(
                self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(
            self.n_channels * self.board_shape[1] * self.board_shape[2],
            int(self.action_n),
        )

        self.critic_cnn = nn.Sequential(
            torch.nn.Conv2d(
                self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.critic_linear = nn.Linear(
            self.n_channels * self.board_shape[1] * self.board_shape[2], 1
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        convolutions = self.network(x) + self.bottleneck(x)

        actor = self.actor_cnn(convolutions)
        actor = actor.reshape(actor.shape[0], -1)
        actor = self.actor_linear(actor)

        critic = self.critic_cnn(convolutions)
        critic = critic.reshape(critic.shape[0], -1)
        critic = self.critic_linear(critic)

        return actor, critic
