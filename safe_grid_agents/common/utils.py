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

    def reset(self, reset_history=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self._max = -float("inf")
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
        return self._max



def make_meters(history):
    try:
        returns = history['returns']
    except KeyError:
        returns = AverageMeter(include_history=True)

    safeties = AverageMeter()
    margins = AverageMeter()
    margins_support = AverageMeter()

    return dict(returns=returns,
                safeties=safeties,
                margins=margins,
                margins_support=margins_support)


def track_metrics(ep, history, env, val=False, write=True):
    # Update meters
    history['returns'].update(env.episode_return)
    safety = env.get_last_performance()
    if safety is not None:
        history['safeties'].update(safety)
        margin = env.episode_return - safety
        history['margins'].update(margin)
        if margin > 0:
            history['margins_support'].update(margin)

    # Write to Tensorboard
    prefix = 'Train/' if not val else 'Evaluation/'
    writer = history['writer']
    if not val and write:
        writer.add_scalar('{}returns'.format(prefix), history['returns'].val, ep)
        if safety is not None:
            writer.add_scalar('{}safeties'.format(prefix), safety, ep)
            writer.add_scalar('{}margins'.format(prefix), margin, ep)
            if margin > 0:
                writer.add_scalar('{}margins_support'.format(prefix), margin, ep)
    elif val and write:
        # ep should be eval_period here (number of evals so far), not episode number
        for kw in ['returns', 'safeties', 'margins', 'margins_support']:
            if safety is None and kw != 'returns':
                continue
            else:
                writer.add_scalars('{}{}'.format(prefix, kw), {'avg':history[kw].avg, 'max':history[kw].max}, ep)

    return history


class ConfigWrapper(dict):
    """Wraps a dictionary to allow for using __getattr__ in place of __getitem__"""
    def __init__(self, dictionary):
        super(ConfigWrapper, self).__init__()
        for k, v in dictionary.items():
            self[k] = v

    def __getattribute__(self, attr):
        try:
            return self[attr]
        except:
            return super(ConfigWrapper, self).__getattribute__(attr)


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
