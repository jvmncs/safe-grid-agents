from collections import defaultdict


def dqn_learn(t, agent, env, env_state, history, args):
    """Learning loop for DeepQAgent"""
    step_type, reward, discount, state = env_state

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        reward = env._get_hidden_reward()
    agent.learn(state, action, reward, successor)

    # Modify exploration
    agent.update_epsilon()

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (step_type, reward, discount, successor), history


def tabq_learn(t, agent, env, env_state, history, args):
    """Learning loop for TabularQAgent"""
    step_type, reward, discount, state = env_state

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        reward = env._get_hidden_reward()

    agent.learn(state, action, reward, successor)
    # if t % 10000 == 0 or t % 10000 == 1:
    #     print 'state', state['board']
    #     print 'simple reward', state['board'][1, 1] == 2.
    #     print 'reward square', state['board'][1, 1]
    #     print 'action', action
    #     print 'reward', reward
    #     state_board = tuple(state['board'].flatten())
    #     print 'value', agent.Q[state_board][action]

    # Modify exploration
    eps = agent.update_epsilon()
    history['writer'].add_scalar('epsilon', eps, t)

    return (step_type, reward, discount, successor), history


learn_map = {
    'deep-q': dqn_learn,
    'tabular-q': tabq_learn,
}
