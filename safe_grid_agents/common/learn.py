from collections import defaultdict
import copy

def dqn_learn(t, agent, env, env_state, history, args):
    """Learning loop for DeepQAgent"""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        reward = env._get_hidden_reward()
    agent.learn(state, action, reward, successor)

    # Modify exploration
    agent.update_epsilon()
    history['writer'].add_scalar('Train/epsilon', agent.epsilon, t)

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (step_type, reward, discount, successor), history


def tabq_learn(t, agent, env, env_state, history, args):
    """Learning loop for TabularQAgent"""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)

    # Act
    action = agent.act_explore(state)
    # if t % 10000 == 0 or t % 10000 == 1 or t % 10000 == 2 or t % 10000 == 3 or t % 10000 == 4:
    #     print('state', state['board'])
    #     print('action', action)
    #     state_board = tuple(state['board'].flatten())
    #     print('value first', agent.Q[state_board])
    #     print('value first', agent.Q[state_board])
    #     print('value first', agent.Q[state_board])
    step_type, reward, discount, successor = env.step(action)

    # if t % 10000 == 0 or t % 10000 == 1 or t % 10000 == 2 or t % 10000 == 3 or t % 10000 == 4:
    #     print(t % 10000)
    #     print('state', state['board'])
    #     print('successor', successor['board'])
    #     print(state['board'] == successor['board'])
    #     print('action again', action)
    #     print('reward', reward)
    #     print('value', agent.Q[state_board])
    #     print('action_explore result', agent.act_explore(state))

    # Learn
    if args.cheat:
        # TODO: fix this, since _get_hidden_reward ßeems to be episodic
        reward = env._get_hidden_reward()

    agent.learn(state, action, reward, successor)
    # if t % 10000 == 0 or t % 10000 == 1 or t % 10000 == 2 or t % 10000 == 3 or t % 10000 == 4:
    #     print('new value', agent.Q[state_board])
    #     print()

    # Modify exploration
    eps = agent.update_epsilon()
    history['writer'].add_scalar('Train/epsilon', eps, t)

    return (step_type, reward, discount, successor), history


learn_map = {
    'deep-q': dqn_learn,
    'tabular-q': tabq_learn,
}
