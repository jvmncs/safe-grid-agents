"""Policy gradient and actor-critic agents."""

from . import base
from ..utils import Rollout

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPO(nn.Module, base.BaseActor, base.BaseLearner, base.BaseExplorer):
    """Actor-critic variant of PPO."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        self.discount = args.discount
        board_shape = env.observation_spec()["board"].shape
        self.n_input = board_shape[0] * board_shape[1]

        # Agent definition
        self.n_layers = args.n_layers
        self.n_hidden = args.n_hidden
        self.horizon = args.horizon
        self.epochs = args.epochs
        self.gae = args.gae_coeff
        self.entropy = args.entropy_bonus
        self.build_ac()
        self.eval()
        self.rollouts = Rollout(states=[], actions=[], rewards=[])

    def act(self, state):
        state_board = torch.tensor(
            state["board"].flatten(),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        p, _ = self.forward(state_board)
        return p.argmax(-1)

    def act_explore(self, state):
        policy = self.policy(state)
        return policy.sample().item()

    def policy(self, state):
        state_board = torch.tensor(
            state["board"].flatten(),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        prepolicy, _ = self.forward(state_board)
        return Categorical(logits=prepolicy)

    def learn(self, state, action, reward, successor):
        # TODO
        raise NotImplementedError

    def build_ac(self):
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

    def forward(self, x):
        x = self.network(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
