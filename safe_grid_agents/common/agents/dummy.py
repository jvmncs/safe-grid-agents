"""Dummy agents for development."""
from safe_grid_agents.common.agents.base import BaseActor

import numpy as np


class RandomAgent(BaseActor):
    """Random walker."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        if args.seed:
            np.random.seed(args.seed)

    def act(self, state):
        return np.random.randint(0, self.action_n)


class SingleActionAgent(BaseActor):
    """Always chooses a single boring action (for testing)."""

    def __init__(self, env, args):
        self.action = args.action
        assert self.action < env.action_spec().maximum + 1, "Not a valid action."

    def act(self, state):
        return self.action
