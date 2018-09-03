import safe_grid_agents.ssrl.warmup as ss
import safe_grid_agents.common.utils as ut
from safe_grid_agents.common.agents import RandomAgent
from collections import defaultdict

def dqn_warmup(agent, env, history, args):
    """Warm start for DQN agent"""
    rando = RandomAgent(env, args) # Exploration only
    print("#### WARMUP ####\n")
    done = True

    for i in range(args.replay_capacity):
        if done:
            history['returns'].update(env.episode_return)
            (step_type, reward, discount, state), done = env.reset(), False

        action = rando.act(None)
        step_type, reward, discount, successor = env.step(action)
        done = step_type.value == 2
        agent.replay.add(state, action, reward, successor)

    return agent, env, history, args

def noop(agent, env, history, args):
    return agent, env, history, args

warmup_map = defaultdict(lambda: noop, {
    'tabular-ssq':ss.random_warmup,
    'deep-q':dqn_warmup,
})
