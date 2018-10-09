"""Agent-specific warmup interactions."""
import safe_grid_agents.ssrl.warmup as ss
from safe_grid_agents.common.agents import RandomAgent
from collections import defaultdict


def dqn_warmup(agent, env, history, args):
    """Warm start for DQN agent."""
    rando = RandomAgent(env, args)  # Exploration only
    print("#### WARMUP ####\n")
    terminal = True

    for _ in range(args.replay_capacity):
        if terminal:
            history["returns"].update(env.episode_return)
            (step_type, reward, discount, state), terminal = env.reset(), False
            board = state["board"]

        action = rando.act(None)
        step_type, reward, discount, successor = env.step(action)
        succ_board = successor["board"]
        terminal = step_type.value == 2
        agent.replay.add(board, action, reward, succ_board, terminal)

    return agent, env, history, args


def noop_warmup(agent, env, history, args):
    """Warm up with noop."""
    return agent, env, history, args


warmup_map = defaultdict(
    lambda: noop_warmup, {"tabular-ssq": ss.random_warmup, "deep-q": dqn_warmup}
)
