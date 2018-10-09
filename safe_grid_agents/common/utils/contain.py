import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import collections


Rollout = collections.namedtuple("Rollout", ["states", "actions", "rewards"])


class ReplayBuffer(object):
    """A buffer to hold past "experiences" for DQN."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, successor):
        self._buffer.append([state, action, reward, successor])

    def sample(self, sample_size):
        ixs = np.random.choice(len(self._buffer), sample_size)
        return [self._buffer[ix] for ix in ixs]
