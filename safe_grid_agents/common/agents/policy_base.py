"""Policy gradient and actor-critic agents."""
from . import base
from .. import utils as ut
from ...types import History, Rollout

import abc
from typing import Tuple
from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOBaseAgent(nn.Module, base.BaseActor, base.BaseLearner, base.BaseExplorer):
    """Actor-critic variant of PPO."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, env, args) -> None:
        super().__init__()
        self.action_n = int(env.action_spec().maximum + 1)
        self.discount = args.discount
        self.board_shape = env.observation_spec()["board"].shape
        self.n_input = self.board_shape[0] * self.board_shape[1]
        self.device = args.device
        self.log_gradients = args.log_gradients

        # Agent definition
        self.lr = args.lr
        self.batch_size = args.batch_size
        # self.horizon = args.horizon
        self.rollouts = args.rollouts
        self.epochs = args.epochs
        self.clipping = args.clipping
        # self.gae = args.gae_coeff
        # self.entropy = args.entropy_bonus

        # Network logistics
        self.build_ac()
        self.to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), self.lr)
        self.old_policy = deepcopy(self)
        self.sync()
        self.old_policy.eval()

    def act(self, state) -> torch.Tensor:
        p, _ = self(state)
        return p.argmax(-1)

    def act_explore(self, state) -> torch.Tensor:
        policy = self.policy(state)
        return policy.sample().item()

    def policy(self, state) -> Categorical:
        prepolicy, _ = self(state)
        return Categorical(logits=prepolicy)

    def learn(self, states, actions, rewards, returns, history, args) -> History:
        states = torch.as_tensor(states, dtype=torch.float, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float, device=self.device)

        for epoch in range(self.epochs):
            rlsz = self.rollouts * states.size(1)
            ixs = torch.randint(rlsz, size=(self.batch_size,), dtype=torch.long)
            s = states.reshape(rlsz, states.shape[2], states.shape[3])[ixs]
            a = actions.reshape(rlsz, -1)[ixs]
            r = returns.reshape(rlsz, -1)[ixs]

            prepolicy, state_values = self(s)
            state_values = state_values.reshape(-1)
            policy_curr = Categorical(logits=prepolicy)

            # Compute critic-adjusted returns
            adv = r - state_values

            # Get log_probs for ratio -- Do not backprop through old policy!
            with torch.no_grad():
                prepolicy, _ = self.old_policy(s)
                log_probs_old = Categorical(logits=prepolicy).log_prob(a)
            log_probs_curr = policy_curr.log_prob(a)
            ratio = torch.exp(log_probs_curr - log_probs_old)

            # Get current policy's entropy
            entropy = policy_curr.entropy().mean()

            # Calculate loss
            vf_loss = nn.functional.mse_loss(state_values, r.squeeze())
            pi_loss = -torch.min(
                (adv * ratio).mean(),
                (adv * ratio.clamp(1 - self.clipping, 1 + self.clipping)).mean(),
            )
            loss = pi_loss + self.critic_coeff * vf_loss - self.entropy_bonus * entropy

            # Logging
            history["writer"].add_scalar(
                "Train/policy_loss", pi_loss.item(), history["t"]
            )
            history["writer"].add_scalar(
                "Train/value_loss", vf_loss.item(), history["t"]
            )
            history["writer"].add_scalar(
                "Train/policy_entropy", self.entropy_bonus * entropy, history["t"]
            )

            # Backprop and step with optional gradient logging
            self.optim.zero_grad()
            loss.backward()
            if self.log_gradients:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        history["writer"].add_histogram(
                            name, param.grad.clone().cpu().data.numpy(), history["t"]
                        )
            self.optim.step()

        return history

    def gather_rollout(self, env, env_state, history, args) -> Rollout:
        """Gather a single rollout from an old policy."""
        step_type, reward, discount, state = env_state
        done = False
        rollout = Rollout(states=[], actions=[], rewards=[], returns=[])

        for r in range(self.rollouts):
            # Rollout loop
            boards, actions, rewards, returns = [], [], [], []
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
                boards.append(board)  # .flatten())
                actions.append(action)
                rewards.append(float(reward))

                state = successor
                history["t"] += 1

            returns = self.get_discounted_returns(rewards)
            history = ut.track_metrics(history, env)
            rollout.states.append(boards)
            rollout.actions.append(actions)
            rollout.rewards.append(rewards)
            rollout.returns.append(returns)

            step_type, reward, discount, state = env.reset()
            done = step_type.value == 2

        return rollout

    def get_discounted_returns(self, rewards) -> torch.Tensor:
        """Compute discounted rewards."""
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        discounted_rewards = [self.discount ** t * r for t, r in enumerate(rewards)]
        cumulative_returns = [
            sum(discounted_rewards[t:]) for t, _ in enumerate(discounted_rewards)
        ]
        return cumulative_returns

    def sync(self) -> None:
        """Sync old and current agent."""
        state_dict = self.state_dict()
        single_state_dict = {
            k: state_dict[k] for k in state_dict.keys() if k[:4] != "old_"
        }
        self.old_policy.load_state_dict(single_state_dict)

    @abc.abstractmethod
    def build_ac(self) -> None:
        """Build the fused actor-critic architecture."""
        return

    @abc.abstractmethod
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return None, None
