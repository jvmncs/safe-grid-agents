"""Dummy agents for development."""
from . import base

import random


class RandomAgent(base.BaseActor):
    """Random walker."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        if args.seed:
            random.seed(args.seed)

    def act(self, state):
        return random.randint(0, self.action_n)


class SingleActionAgent(base.BaseActor):
    """Always chooses a single boring action (for testing)."""

    def __init__(self, env, args):
        self.action = args.action
        assert self.action < env.action_spec().maximum + 1, "Not a valid action."

    def act(self, state):
        return self.action
