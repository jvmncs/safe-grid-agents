from collections import defaultdict
import copy


def dqn_learn(t, agent, env, env_state, history, args):
    """Learning loop for DeepQAgent."""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        # TODO: fix this, since _get_hidden_reward seems to be episodic
        reward = env._get_hidden_reward()
    loss = agent.learn(state, action, reward, successor)
    history["writer"].add_scalar("Train/loss", loss, t)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (step_type, reward, discount, successor), history


def tabq_learn(t, agent, env, env_state, history, args):
    """Learning loop for TabularQAgent."""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        # TODO: fix this, since _get_hidden_reward seems to be episodic
        reward = env._get_hidden_reward()
    agent.learn(state, action, reward, successor)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    return (step_type, reward, discount, successor), history


learn_map = {"deep-q": dqn_learn, "tabular-q": tabq_learn}
