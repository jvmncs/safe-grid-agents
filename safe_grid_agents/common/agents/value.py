"""Value-based agents."""
from . import base
from .. import utils
from ...types import History, ExperienceBatch

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Baseline agents
class TabularQAgent(base.BaseActor, base.BaseLearner, base.BaseExplorer):
    """Tabular Q-learner."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        self.discount = args.discount

        # Agent definition
        self.future_eps = [
            1.0 - (1 - args.epsilon) * t / args.epsilon_anneal
            for t in range(args.epsilon_anneal)
        ]
        self.update_epsilon()
        self.epsilon = 0.0
        self.discount = args.discount
        self.lr = args.lr
        self.Q = defaultdict(lambda: np.zeros(self.action_n))

    def act(self, state):
        state_board = tuple(state.flatten())
        return np.argmax(self.Q[state_board])

    def act_explore(self, state):
        if np.random.sample() < self.epsilon:
            action = np.random.choice(self.action_n)
        else:
            action = self.act(state)
        return action

    def learn(self, state, action, reward, successor):
        """Q learning."""
        state_board = tuple(state.flatten())
        successor_board = tuple(successor.flatten())
        action_next = self.act(successor)
        value_estimate_next = self.Q[successor_board][action_next]
        target = reward + self.discount * value_estimate_next
        differential = target - self.Q[state_board][action]
        self.Q[state_board][action] += self.lr * differential

    def update_epsilon(self):
        """Update epsilon exploration constant."""
        if len(self.future_eps) > 0:
            self.epsilon = self.future_eps.pop(0)
        return self.epsilon


class DeepQAgent(base.BaseActor, base.BaseLearner, base.BaseExplorer):
    """Q-learner with deep function approximation."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        board_shape = env.observation_spec()["board"].shape
        self.n_input = board_shape[0] * board_shape[1]
        self.device = args.device
        self.log_gradients = args.log_gradients

        # Agent definition
        self.future_eps = [
            1.0 - (1 - args.epsilon) * t / args.epsilon_anneal
            for t in range(args.epsilon_anneal)
        ]
        self.update_epsilon()
        self.discount = args.discount
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.Q = self.build_Q(self.n_input, args.n_layers, args.n_hidden)
        self.Q.eval()
        self.Q.to(self.device)
        self.target_Q = self.build_Q(self.n_input, args.n_layers, args.n_hidden)
        self.target_Q.eval()
        self.target_Q.to(self.device)
        self.replay = utils.ReplayBuffer(args.replay_capacity)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=args.lr, amsgrad=True)

    def act(self, state):
        state_board = self._lift(state.flatten()).reshape(1, -1)
        scores = self.Q(state_board)
        return scores.argmax(1)

    def act_explore(self, state):
        policy = self.policy(state)
        return policy.sample().item()

    def policy(self, state):
        """Produce the entire policy distribution over actions for a state."""
        argmax = self.act(state)
        probs = (
            torch.zeros(
                self.action_n,
                requires_grad=False,
                dtype=torch.float32,
                device=self.device,
            )
            + self.epsilon / self.action_n
        )
        probs[argmax] += 1 - self.epsilon
        return Categorical(probs=probs)

    def learn(self, state, action, reward, successor, terminal, history) -> History:
        self.replay.add(state, action, reward, successor, terminal)
        states, actions, rewards, successors, terminals = self.process(
            self.replay.sample(self.batch_size)
        )
        self.Q.train()
        Qs = self.Q(states).gather(1, actions)
        next_Qs = self.target_Q(successors).max(1)[0]
        next_Qs[terminals] = 0
        expected_Qs = self.discount * next_Qs + rewards
        loss = F.mse_loss(Qs, expected_Qs)
        history["writer"].add_scalar("Train/value_loss", loss.item(), history["t"])

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), 10.0)
        if self.log_gradients:
            for name, param in self.Q.named_parameters():
                history["writer"].add_histogram(
                    name, param.grad.clone().cpu().data.numpy(), history["t"]
                )
        self.optim.step()
        self.Q.eval()
        return history

    def sync_target_Q(self) -> None:
        """Sync target network with most recent behavior network."""
        self.target_Q.load_state_dict(self.Q.state_dict())

    def update_epsilon(self) -> float:
        """Update epsilon exploration constant."""
        if len(self.future_eps) > 0:
            self.epsilon = self.future_eps.pop(0)
        return self.epsilon

    def build_Q(self, n_input: int, n_layers: int, n_hidden: int) -> nn.Sequential:
        """Build a single Q network."""
        first = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU())
        hidden = nn.Sequential(
            *tuple(
                nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU())
                for _ in range(n_layers - 1)
            )
        )
        last = nn.Linear(n_hidden, int(self.action_n))
        return nn.Sequential(first, hidden, last)

    def process(self, experiences) -> ExperienceBatch:
        """Convert gridworld representations to torch Tensors."""
        boards = np.concatenate([experience.state.flatten() for experience in experiences], axis=0)
        boards = self._lift(boards, grad=True).reshape(-1, self.n_input)

        actions = [experience.action for experience in experiences]
        actions = self._lift(actions, dtype=torch.long).reshape(-1, 1)

        rewards = [experience.reward for experience in experiences]
        rewards = self._lift(rewards)

        successors = np.concatenate([experience.successor.flatten() for experience in experiences], axis=0)
        successors = self._lift(successors, grad=True).reshape(-1, self.n_input)

        terminals = [experience.terminal for experience in experiences]
        terminals = self._lift(terminals, dtype=torch.uint8)

        return boards, actions, rewards, successors, terminals

    def _lift(self, x, dtype=torch.float32, grad=False):
        return torch.as_tensor(x, dtype=dtype, device=self.device, requires_grad=grad)
