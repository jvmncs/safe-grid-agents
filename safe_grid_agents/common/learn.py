"""Agent-specific learning interactions."""
import copy


def dqn_learn(agent, env, env_state, history, args):
    """Learning loop for DeepQAgent."""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)  # Make sure `state` doesn't change next step
    board = state["board"]
    t = history["t"]

    # Act
    action = agent.act_explore(board)
    step_type, reward, discount, successor = env.step(action)
    terminal = env_state[0].value == 2
    succ_board = successor["board"]

    # Learn
    if args.cheat:
        # TODO: fix this, since _get_hidden_reward seems to be episodic
        reward = env._get_hidden_reward()
        # In case the agent is drunk, use the actual action they took
        try:
            action = successor["extra_observations"]["actual_actions"]
        except KeyError:
            pass
    history = agent.learn(board, action, reward, succ_board, terminal, history)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (step_type, reward, discount, successor), history


def tabq_learn(agent, env, env_state, history, args):
    """Learning loop for TabularQAgent."""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)
    t = history["t"]

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
