"""Utilities dealing with meters and tracking metrics."""

import bisect
import numpy as np


class AverageMeter(object):
    """Compute and store the average and current value.

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
        returns = history["returns"]
    except KeyError:
        returns = AverageMeter(include_history=True)

    safeties = AverageMeter()
    margins = AverageMeter()
    margins_support = AverageMeter()

    return dict(
        returns=returns,
        safeties=safeties,
        margins=margins,
        margins_support=margins_support,
    )


def track_metrics(history, env, eval=False, write=True):
    if hasattr(env, "_env"):
        _env = env._env
    else:
        _env = env
    # Update meters
    if not eval:
        ep = history["episode"]
    else:
        ep = history["period"]
    history["returns"].update(_env.episode_return)
    safety = _env.get_last_performance()
    if safety is not None:
        history["safeties"].update(safety)
        margin = _env.episode_return - safety
        history["margins"].update(margin)
        if margin > 0:
            history["margins_support"].update(margin)

    # Write to Tensorboard
    if write:
        prefix = "Train/" if not eval else "Evaluation/"
        writer = history["writer"]
        if not eval:
            writer.add_scalar("{}returns".format(prefix), history["returns"].val, ep)
            if safety is not None:
                writer.add_scalar("{}safeties".format(prefix), safety, ep)
                writer.add_scalar("{}margins".format(prefix), margin, ep)
                if margin > 0:
                    writer.add_scalar("{}margins_support".format(prefix), margin, ep)
        else:
            # ep should be eval_period here (number of evals so far), not episode number
            for kw in ["returns", "safeties", "margins", "margins_support"]:
                if safety is None and kw != "returns":
                    continue
                else:
                    writer.add_scalars(
                        "{}{}".format(prefix, kw),
                        {"avg": history[kw].avg, "max": history[kw].max},
                        ep,
                    )

    return history
