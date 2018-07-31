import bisect
import collections
import numpy as np
import random


class AverageMeter(object):
    """Computes and stores the average and current value
    Extended from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, include_history=False):
        self.include_history = include_history
        self.reset(reset_history=True)
        self._max = None

    def reset(self, reset_history=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self._max = -float('inf')
        self.count = 0
        if reset_history:
            self._history = None
            if self.include_history:
                self._history = []
        
    def update(self, val, n=1):
        self.val = val
        self._max = max(self._max, self.val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self._history is not None:
            for x in range(n):
                bisect.insort(self._history, val)

    def quantile(self, delta):
        if self._history is not None:
            q = (1 - delta) * 100
            return np.percentile(self._history, q=q)
        else:
            raise RuntimeError("Meter instantiated without history.")

    @property
    def max(self):
        if (self._history is None or len(self._history) == 0):
            if self._max == -float("inf"):
                raise RuntimeError("History empty.")
        else:
            return self._max

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
        