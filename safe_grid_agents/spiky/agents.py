"""PPO Agent for CRMDPs."""
import torch
import numpy as np
from typing import Generator, List

from safe_grid_agents.common.utils import track_metrics
from safe_grid_agents.common.agents.policy_cnn import PPOCNNAgent
from safe_grid_agents.types import Rollout

from ai_safety_gridworlds.environments.tomato_crmdp import REWARD_FACTOR


def _get_agent_position(board, agent_value):
    x_pos, y_pos = np.unravel_index(
        np.argwhere(np.ravel(board) == agent_value), board.shape
    )
    x_pos, y_pos = x_pos.flat[0], y_pos.flat[0]
    return x_pos, y_pos


def _manhatten_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def d_tomato_crmdp(X, Y):
    assert X.shape == Y.shape
    return REWARD_FACTOR * np.sum(X != Y)


def d_toy_gridworlds(X, Y):
    assert X.shape == Y.shape
    X = X[0, ...]
    Y = Y[0, ...]
    # toy gridworlds use value 0 to denote the agent on the board
    X_pos_x, X_pos_y = _get_agent_position(X, agent_value=0)
    Y_pos_x, Y_pos_y = _get_agent_position(Y, agent_value=0)
    return _manhatten_distance(X_pos_x, Y_pos_x, X_pos_y, Y_pos_y)


def d_trans_boat(X, Y):
    assert X.shape == Y.shape
    X_initial, X_final = X[0, ...], X[1, ...]
    Y_initial, Y_final = Y[0, ...], Y[1, ...]

    # deepmind gridworlds use value 2 to denote the agent on the board
    X_initial_pos_x, X_initial_pos_y = _get_agent_position(X_initial, agent_value=2)
    Y_initial_pos_x, Y_initial_pos_y = _get_agent_position(Y_initial, agent_value=2)
    X_final_pos_x, X_final_pos_y = _get_agent_position(X_final, agent_value=2)
    Y_final_pos_x, Y_final_pos_y = _get_agent_position(Y_final, agent_value=2)

    X_direction_x = X_final_pos_x - X_initial_pos_x
    X_direction_y = X_final_pos_y - X_initial_pos_y
    Y_direction_x = Y_final_pos_x - Y_initial_pos_x
    Y_direction_y = Y_final_pos_y - Y_initial_pos_y

    initial_position_distance = _manhatten_distance(
        X_initial_pos_x, Y_initial_pos_x, X_initial_pos_y, Y_initial_pos_y
    )
    direction_distance = int(X_direction_x != Y_direction_x)
    direction_distance += int(X_direction_y != Y_direction_y)

    return initial_position_distance + direction_distance


ENV_TO_D = {
    "corners": d_toy_gridworlds,
    "way": d_toy_gridworlds,
    "tomato-crmdp": d_tomato_crmdp,
    "trans-boat": d_trans_boat,
}


class PPOCRMDPAgent(PPOCNNAgent):
    """PPO Agent for CRMDPs."""

    def __init__(self, env, args) -> None:
        super().__init__(env, args)
        self.states = dict()
        self.d = ENV_TO_D[args.env_alias]
        self.epsilon = 1e-3
        self.rllb = dict()

    def _mark_state_corrupt(self, board, reward) -> None:
        assert board.dtype == np.float32
        self.states[board.tostring()] = [False, reward]

    def _mark_state_safe(self, board, reward) -> None:
        assert board.dtype == np.float32
        self.states[board.tostring()] = [True, reward]

    def _is_state_corrupt(self, board) -> bool:
        if board.tostring() in self.states:
            return not self.states[board.tostring()][0]
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
        for corrupt_board, corrupt_reward in self._iterate_corrupt_states():
            rllb = None
            for safe_board, safe_reward in self._iterate_safe_states():
                bound = safe_reward - self.d(safe_board, corrupt_board)
                if rllb is None or bound > rllb:
                    rllb = bound
            self.rllb[corrupt_board.tostring()] = rllb

    def _get_TLV(self, boardX, rewardX, state_iterator) -> float:
        """Return the total Lipschitz violation of a state X w.r.t a set of states.
        Each state is only added once to the TLV."""
        TLV = 0
        unique_states = set()
        for boardY, rewardY in state_iterator:
            if boardY.tostring() not in unique_states:
                TLV += max(0, abs(rewardX - rewardY) - self.d(boardY, boardX))
                unique_states.add(boardY.tostring())
        return TLV

    def get_modified_reward(self, board, reward) -> float:
        """Return the reward to use for optimizing the policy based on the rllb."""
        if self._is_state_corrupt(board):
            return self.rllb[board.tostring()]
        else:
            return reward

    def get_modified_rewards_for_rollout(self, boards, rewards) -> List[float]:
        """
        Returns a list of rewards for a given rollout that has been updated based
        on the rllb.
        """
        new_rewards = []
        for i in range(len(rewards)):
            new_rewards.append(self.get_modified_reward(boards[i], rewards[i]))
        return new_rewards

    def identify_corruption_in_trajectory(self, boards, rewards) -> None:
        """Perform detection of corrupt states on a trajectory.

        Updates the set of safe states and corrupt states with all new states,
        that are being visited in this trajectory. Then updates the self.rllb
        dict, so that we can get the modified reward function.
        """
        boards = np.array(boards)
        rewards = np.array(rewards)

        TLV = np.zeros(len(boards))
        for i in range(len(boards)):
            TLV[i] = self._get_TLV(boards[i], rewards[i], zip(boards, rewards))

        TLV_sort_idx = np.argsort(TLV)[::-1]
        non_corrupt_idx = list(range(len(boards)))
        added_corrupt_states = False

        # iterate over all states in the trajectory in order decreasing by their TLV
        for i in range(len(boards)):
            idx = TLV_sort_idx[i]
            if not added_corrupt_states:
                # performance improvement
                new_TLV = TLV[idx]
            else:
                new_TLV = self._get_TLV(
                    boards[idx],
                    rewards[idx],
                    zip(boards[non_corrupt_idx], rewards[non_corrupt_idx]),
                )

            if new_TLV <= self.epsilon:
                if not self._is_state_corrupt(boards[idx]):
                    self._mark_state_safe(boards[idx], rewards[idx])
                break
            else:
                self._mark_state_corrupt(boards[idx], rewards[idx])
                non_corrupt_idx.remove(idx)
                added_corrupt_states = True

        if added_corrupt_states:
            self._update_rllb()

    def gather_rollout(self, env, env_state, history, args) -> Rollout:
        """Gather a single rollout from an old policy.

        Based on the gather_rollout function of the regular PPO agents.
        This version also tracks the successor states of each action.
        Based on this the corrupted states can be detected before performing
        the training step."""
        state, reward, done, info = env_state
        done = False
        rollout = Rollout(states=[], actions=[], rewards=[], returns=[])
        successors = []

        for r in range(self.rollouts):
            successors_r = []
            # Rollout loop
            states, actions, rewards, returns = [], [], [], []
            while not done:
                with torch.no_grad():
                    action = self.old_policy.act_explore(state)
                    successor, reward, done, info = env.step(action)

                # Maybe cheat
                if args.cheat:
                    reward = info["hidden_reward"]
                    # In case the agent is drunk, use the actual action they took
                    try:
                        action = info["extra_observations"]["actual_actions"]
                    except KeyError:
                        pass

                # Store data from experience
                states.append(state)  # .flatten())
                actions.append(action)
                rewards.append(float(reward))
                successors_r.append(successor)

                state = successor
                history["t"] += 1

            if r != 0:
                history["episode"] += 1

            self.identify_corruption_in_trajectory(successors_r, rewards)
            rewards = self.get_modified_rewards_for_rollout(successors_r, rewards)

            returns = self.get_discounted_returns(rewards)
            history = track_metrics(history, env)
            rollout.states.append(states)
            rollout.actions.append(actions)
            rollout.rewards.append(rewards)
            rollout.returns.append(returns)
            successors.append(successors_r)

            state = env.reset()
            done = False

        return rollout
