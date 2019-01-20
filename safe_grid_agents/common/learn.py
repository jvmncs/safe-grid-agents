"""Agent-specific learning interactions."""
import copy
import functools

from safe_grid_agents.common import utils as ut


def whiler(function):
    """Evaluate the agent-specific learn function `fn` inside of a generic while loop."""

    @functools.wraps(function)
    def stepbystep(agent, env, env_state, history, args):
        done = False
        eval_next = False
        while not done:
            env_state, history = function(agent, env, env_state, history, args)
            done = env_state[2]
            history["t"] += 1
        history = ut.track_metrics(history, env)
        if history["episode"] % args.eval_every == args.eval_every - 1:
            eval_next = True

        return env_state, history, eval_next

    return stepbystep


@whiler
def dqn_learn(agent, env, env_state, history, args):
    """Learning loop for DeepQAgent."""
    state, reward, done, info = env_state

    t = history["t"]

    # Act
    action = agent.act_explore(state)
    successor, reward, done, info = env.step(action)

    # Learn
    if args.cheat:
        reward = info["hidden_reward"]
        # In case the agent is drunk, use the actual action they took
        try:
            action = info["extra_observations"]["actual_actions"]
        except KeyError:
            pass
    history = agent.learn(state, action, reward, successor, done, history)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (successor, reward, done, info), history


@whiler
def tabq_learn(agent, env, env_state, history, args):
    """Learning loop for TabularQAgent."""
    state, reward, done, info = env_state
    t = history["t"]

    # Act
    action = agent.act_explore(state)
    successor, reward, done, info = env.step(action)

    # Learn
    if args.cheat:
        reward = info["hidden_reward"]
        # In case the agent is drunk, use the actual action they took
        try:
            action = info["extra_observations"]["actual_actions"]
        except KeyError:
            pass
    agent.learn(state, action, reward, successor)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    return (successor, reward, done, info), history


def ppo_learn(agent, env, env_state, history, args):
    """Learning loop for PPOAgent."""
    eval_next = False
    # Act
    rollout = agent.gather_rollout(env, env_state, history, args)

    # Learn
    history = agent.learn(*rollout, history, args)

    # Sync old and current policy
    agent.sync()

    # Check for evaluating next
    if history["episode"] % args.eval_every == args.eval_every - 1:
        eval_next = True

    return env_state, history, eval_next


learn_map = {
    "deep-q": dqn_learn,
    "tabular-q": tabq_learn,
    "ppo-mlp": ppo_learn,
    "ppo-cnn": ppo_learn,
    "ppo-crmdp": ppo_learn,
}
