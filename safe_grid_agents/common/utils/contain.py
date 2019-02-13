"""Containers useful for storing agent experiences."""
import numpy as np
import collections
from typing import List
from safe_grid_agents.types import Experience


class ReplayBuffer:
    """A buffer to hold past "experiences" for DQN."""

    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, successor, terminal) -> None:
        """Add to experience replay buffer."""
        self._buffer.append(Experience(state, action, reward, successor, terminal))

    def sample(self, sample_size) -> List[Experience]:
        """Sample experience from replay buffer."""
        ixs = np.random.choice(len(self._buffer), sample_size)
        return [self._buffer[ix] for ix in ixs]
