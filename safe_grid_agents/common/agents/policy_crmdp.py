"""PPO Agent for CRMDPs."""
import sys
import numpy as np
from typing import Generator

from .policy_cnn import PPOCNNAgent
from ...types import Rollout

sys.path.insert(0, "ai-safety-gridworlds/")
from ai_safety_gridworlds.environments.tomato_crmdp import REWARD_FACTOR


class PPOCRMDPAgent(PPOCNNAgent):
    """PPO Agent for CRMDPs."""

    def __init__(self, env, args) -> None:
        super().__init__(env, args)
        self.states = dict()
        self.d = lambda x, y: REWARD_FACTOR * np.sum(x != y)
        self.epsilon = 1e-3
        self.rllb = dict()

    def _mark_state_corrupt(self, board, reward) -> None:
        self.states[board.tostring()] = [False, reward]

    def _mark_state_safe(self, board, reward) -> None:
        self.states[board.tostring()] = [True, reward]

    def _is_state_corrupt(self, board) -> bool:
        if board.tostring() in self.states:
            return self.states[board.tostring()][0]
        else:
            return False

    def _iterate_safe_states(self) -> Generator[np.array, None, None]:
        for board_str in self.states.keys():
            if self.states[board_str][0]:
                board = np.fromstring(board_str, dtype=np.float32, count=self.n_input)
                board = np.reshape(board, self.board_shape)
                yield board, self.states[board_str][1]

    def _iterate_corrupt_states(self) -> Generator[np.array, None, None]:
        for board_str in self.states.keys():
            if not self.states[board_str][0]:
                board = np.fromstring(board_str, dtype=np.float32, count=self.n_input)
                board = np.reshape(board, self.board_shape)
                yield board, self.states[board_str][1]

    def _update_rllb(self) -> None:
        """Update the reward lower Lipschitz bound."""
        iterator_safe = self._iterate_safe_states()
        iterator_corrupt = self._iterate_corrupt_states()

        for corrupt_board, corrupt_reward in iterator_corrupt:
            rllb = None
            for safe_board, safe_reward in iterator_safe:
                bound = safe_reward - self.d(safe_board, corrupt_board)
                if rllb is None or bound > rllb:
                    rllb = bound
            self.rllb[corrupt_board.tostring()] = rllb

    def _get_TLV(self, boardX, rewardX, state_iterator) -> float:
        """Return the total Lipschitz violation of a state X w.r.t a set of states."""
        TLV = 0
        for boardY, rewardY in state_iterator:
            TLV += abs(rewardX - rewardY) - self.d(boardY, boardX)
        return TLV

    def get_modified_reward(self, board, reward) -> float:
        """Return the reward to use for optimizing the policy based on the rllb."""
        if self._is_state_corrupt(board):
            return self.rllb[board]
        else:
            return reward

    def identify_corruption_in_trajectory(self, boards, rewards) -> None:
        """Perform detection of corrupt states on a trajectory.

        Updates the set of safe states and corrupt states with all new states,
        that are being visited in this trajectory. Then updates the self.rllb
        dict, so that we can get the modified reward function.
        """
        trajectory_iterator = zip(boards, rewards)

        TLV = np.zeros(len(boards))
        for i in range(len(boards)):
            TLV[i] = self._get_TLV(boards[i], rewards[i], trajectory_iterator)

        TLV_sort_idx = np.argsort(TLV)[::-1]
        added_corrupt_states = False

        # iterate over all states in the trajectory in order decreasing by their TLV
        for i in range(len(boards)):
            idx = TLV_sort_idx[i]
            if not added_corrupt_states:
                # performance improvement
                new_TLV = TLV[idx]
            else:
                new_TLV = self._get_TLV(boards[idx], rewards[idx], trajectory_iterator)

            if new_TLV <= self.epsilon:
                break
            else:
                self._mark_state_corrupt(boards[idx], rewards[idx])
                added_corrupt_states = True

        if added_corrupt_states:
            self._update_rllb()

    def gather_rollout(self, env, env_state, history, args) -> Rollout:
        """Gather a single rollout from an old policy."""
        rollouts = super().gather_rollout(env, env_state, history, args)

        for r in range(self.rollouts):
            self.identify_corruption_in_trajectory(
                rollouts.states[r], rollouts.rewards[r]
            )
            rollouts.rewards[r] = [
                self.get_modified_reward(board, reward)
                for board, reward in zip(rollouts.states[r], rollouts.rewards[r])
            ]

        return rollouts
