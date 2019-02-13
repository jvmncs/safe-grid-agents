"""Agent-specific warmup interactions."""
from collections import defaultdict

import safe_grid_agents.ssrl.warmup as ss
from safe_grid_agents.common.agents import RandomAgent


def dqn_warmup(agent, env, history, args):
    """Warm start for DQN agent."""
    rando = RandomAgent(env, args)  # Exploration only
    print("#### WARMUP ####\n")
    done = True

    for _ in range(args.replay_capacity):
        if done:
            history["returns"].update(env._env.episode_return)
            state, done = env.reset(), False

        action = rando.act(None)
        successor, reward, done, _ = env.step(action)
        agent.replay.add(state, action, reward, successor, done)

    return agent, env, history, args


def noop_warmup(agent, env, history, args):
    """Warm up with noop."""
    return agent, env, history, args


WARMUP_MAP = defaultdict(
    lambda: noop_warmup, {"tabular-ssq": ss.random_warmup, "deep-q": dqn_warmup}
)
