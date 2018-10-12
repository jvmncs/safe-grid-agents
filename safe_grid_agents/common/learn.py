"""Agent-specific learning interactions."""
import copy
import functools


def whiler(function):
    """Evaluate the agent-specific learn function `fn` inside of a generic while loop."""

    @functools.wraps(function)
    def stepbystep(agent, env, env_state, history, args):
        done = False
        eval_next = False
        while not done:
            env_state, history = function(agent, env, env_state, history, args)
            done = env_state[0].value == 2
            if history["t"] % args.eval_every == args.eval_every - 1:
                eval_next = True
        return env_state, history, eval_next

    return stepbystep


@whiler
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
        current_score = env._get_hidden_reward()
        reward = current_score - history["last_score"]
        history["last_score"] = current_score
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

    # Increment timestep for future tracking
    history["t"] += 1
    if t % args.eval_every == args.eval_every - 1:
        eval_next = True

    return (step_type, reward, discount, successor), history, eval_next


@whiler
def tabq_learn(agent, env, env_state, history, args):
    """Learning loop for TabularQAgent."""
    step_type, reward, discount, state = env_state
    state = copy.deepcopy(state)
    board = state["board"]
    t = history["t"]

    # Act
    action = agent.act_explore(board)
    step_type, reward, discount, successor = env.step(action)
    succ_board = successor["board"]

    # Learn
    if args.cheat:
        current_score = env._get_hidden_reward()
        reward = current_score - history["last_score"]
        history["last_score"] = current_score
        # In case the agent is drunk, use the actual action they took
        try:
            action = successor["extra_observations"]["actual_actions"]
        except KeyError:
            pass
    agent.learn(board, action, reward, succ_board)

    # Modify exploration
    eps = agent.update_epsilon()
    history["writer"].add_scalar("Train/epsilon", eps, t)

    # Increment timestep for future tracking
    history["t"] += 1
    if t % args.eval_every == args.eval_every - 1:
        eval_next = True

    return (step_type, reward, discount, successor), history, eval_next


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
    if history["t"] % args.eval_every == args.eval_every - 1:
        eval_next = True

    return env_state, history, eval_next


learn_map = {"deep-q": dqn_learn, "tabular-q": tabq_learn, "ppo": ppo_learn}
