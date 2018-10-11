"""Policy gradient and actor-critic agents."""
from . import base
from ...types import History, Rollout

from typing import Tuple
from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent(nn.Module, base.BaseActor, base.BaseLearner, base.BaseExplorer):
    """Actor-critic variant of PPO."""

    def __init__(self, env, args) -> None:
        super().__init__()
        self.action_n = int(env.action_spec().maximum + 1)
        self.discount = args.discount
        board_shape = env.observation_spec()["board"].shape
        self.n_input = board_shape[0] * board_shape[1]
        self.device = args.device
        self.log_gradients = args.log_gradients

        # Agent definition
        self.lr = args.lr
        self.n_layers = args.n_layers
        self.n_hidden = args.n_hidden
        # self.horizon = args.horizon
        # self.epochs = args.epochs
        self.clipping = args.clipping
        # self.gae = args.gae_coeff
        # self.entropy = args.entropy_bonus
        self.build_ac()
        self.optim = torch.optim.Adam(self.parameters(), self.lr)
        self.old_policy = deepcopy(self)
        self.sync()
        self.old_policy.eval()

    def act(self, state) -> torch.Tensor:
        state_board = torch.tensor(
            state["board"].flatten(),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        p, _ = self.forward(state_board)
        return p.argmax(-1)

    def act_explore(self, state) -> torch.Tensor:
        policy = self.policy(state)
        return policy.sample().item()

    def policy(self, state) -> Categorical:
        state_board = torch.tensor(
            state.flatten(),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        prepolicy, _ = self(state_board)
        return Categorical(logits=prepolicy)

    def learn(self, states, actions, rewards, history, args) -> History:
        states = torch.as_tensor(states, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long)

        cumulative_returns = self.get_discounted_returns(rewards)

        prepolicy, state_values = self(states)
        state_values = state_values.reshape(-1)
        policy_curr = Categorical(logits=prepolicy)

        # Compute critic-adjusted returns
        adv = cumulative_returns - state_values
        # Update VF
        vf_loss = nn.functional.mse_loss(state_values, cumulative_returns)

        # Old model is copied anyway, so no updates to it are necessary.
        with torch.no_grad():
            prepolicy, _ = self.old_policy(states)
            log_probs_old = Categorical(logits=prepolicy).log_prob(actions)
        log_probs_curr = policy_curr.log_prob(actions)

        ratio = torch.exp(log_probs_curr - log_probs_old)

        policy_loss = torch.min(
            -(adv * ratio).mean(),
            -(adv * ratio.clamp(1 - self.clipping, 1 + self.clipping)).mean(),
        )
        loss = policy_loss + vf_loss
        history["writer"].add_scalars(
            "Train/{}",
            {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": vf_loss.item(),
            },
            history["t"],
        )

        self.optim.zero_grad()
        loss.backward()
        if self.log_gradients:
            for name, param in self.named_parameters():
                history["writer"].add_histogram(
                    name, param.grad.clone().cpu().data.numpy(), history["t"]
                )
        self.optim.step()

        history["t"] += 1
        return history

    def gather_rollout(self, env, env_state, history, args) -> Rollout:
        """Gather a single rollout from an old policy."""
        step_type, reward, discount, state = env_state
        done = False
        rollout = Rollout(states=[], actions=[], rewards=[])

        # Rollout loop
        while not done:
            state = deepcopy(state)
            board = state["board"]
            action = self.old_policy.act_explore(board)
            with torch.no_grad():
                step_type, reward, discount, successor = env.step(action)
                done = step_type.value == 2

            # Maybe cheat
            if args.cheat:
                current_score = env._get_hidden_reward()
                reward = current_score - history["last_score"]
                history["last_score"] = current_score
                # In case the agent is drunk, use the actual action they took
                try:
                    action = successor["extra_observations"]["actual_actions"]
                except KeyError:
                    pass

            # Store data from experience
            rollout.states.append(board.flatten())
            rollout.actions.append(action)
            rollout.rewards.append(reward)

            state = successor

        return rollout

    def get_discounted_returns(self, rewards) -> torch.Tensor:
        """Compute discounted rewards."""
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        discounted_rewards = [self.discount ** t * r for t, r in enumerate(rewards)]
        cumulative_returns = torch.as_tensor(
            [sum(discounted_rewards[t:]) for t, _ in enumerate(discounted_rewards)],
            dtype=torch.float,
            device=self.device,
        )
        return cumulative_returns

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
        x = self.network(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

    def sync(self) -> None:
        """Sync old and current agent."""
        state_dict = self.state_dict()
        single_state_dict = {
            k: state_dict[k] for k in state_dict.keys() if k[:4] != "old_"
        }
        self.old_policy.load_state_dict(single_state_dict)
