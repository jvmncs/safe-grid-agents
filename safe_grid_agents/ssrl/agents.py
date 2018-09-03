# Semi-supervised agents

import safe_grid_agents.common.agents as agents

from collections import defaultdict
import numpy as np


class TabularSSQAgent(agents.TabularQAgent):
    """
    A tabular Q-learner for SSRL in CRMDPs.
    Loosely inspired by the quantilising agent from Everitt et al. 2017,
    but with a Bayesian estimate of the corruption C.
    """
    def __init__(self, env, args):
        super(self.__class__, self).__init__(env, args)

        # SSRL
        self.budget = args.budget
        self.C = defaultdict(lambda: args.C_prior)
        self.C_support = defaultdict(lambda: 0) # only over visited states, value is number of times visited
        self.corrupt_episodes = 0
        self.episodes = 0
        self.reset_history(False, False) # Corruption learning needs episode history

    def act_explore(self, state):
        action = super(self.__class__, self).act_explore(state)
        self._history.append(state)
        return action

    def learn(self, state, action, reward, successor):
        """Q learning with corruption map"""
        state_board = tuple(state['board'].flatten())
        successor_board = tuple(successor['board'].flatten())
        reward_estimate = reward * (1 - self.C[state_board])
        action_next = self.act(successor_board)
        target = reward_estimate + self.discount * self.Q[successor_board][action_next]
        differential = target - self.Q[state_board][action]
        self.Q[state_board][action] += self.lr * differential

    # Semi-supervised corruption learning
    def query_H(self, env):
        """Query H, a more informed agent"""
        self.budget -= 1
        return env.get_last_performance()

    def learn_C(self, corrupt_episode):
        """Learn probability of being in a corrupt state (Bayesian update)"""
        for state in self._history:
            state_board = tuple(state['board'].flatten())
            if not corrupt_episode:
                self.C[state_board] *= 0 # P(C | ~E) = 0
                try:
                    del self.C_support[state_board]
                except KeyError:
                    pass
            else:
                # Add state to support
                self.C_support[state_board] += 1

                # Uses pessimistic estimator of P(E), since P(E) is actually expectaton of hitting a
                # corrupt episode under the current policy, which should learn to avoid corruption
                # fairly quickly
                # self.C[state] := P(C | E), probability that state is corrupt given the episode is
                #   == P(E | C) * P(C) / P(E)
                #   == 1 * P(C) / P(E), P(E | C) = 1 since episode is corrupt if any state is
                #   == self.C[state] / P(E)
                #   =~ self.C[state] / (N(corrupt_episodes)/N(episodes))
                #   == self.C[state] * N(episodes) / N(corrupt_episodes)

                self.C[state_board]  *= self.episodes / (self.corrupt_episodes + 1)
        self.reset_history(corrupt_episode)

    def reset_history(self, corrupt, increment_episode=True):
        if corrupt:
            self.corrupt_episodes += 1
        if increment_episode:
            self.episodes += 1
        self._history = []

    def update_global_prior(self):
        # (Current expectation of q/|state_space| based on previously visited states)
        self.C.default_factory = lambda: len(self.C_support)/len(self.C)
