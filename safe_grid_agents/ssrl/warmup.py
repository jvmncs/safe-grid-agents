from safe_grid_agents.common.agents import RandomAgent
import safe_grid_agents.common.utils as ut

def random_warmup(agent, env, history, args):
    """Warm start for SSRL agent"""
    returns = history['returns']
    safeties = history['safeties']
    margins = history['margins']
    margins_support = history['margins_support']
    rando = RandomAgent(env, args) # Exploration only
    print("#### WARMUP ####\n")
    warmup_phase = int(args.budget * args.warmup)
    init_budget = args.budget
    t = 0
    while t < warmup_phase:
        (step_type, reward, discount, state), done = env.reset(), False
        while not done:
            action = rando.act(None)
            step_type, reward, discount, successor = env.step(action)
            done = step_type.value == 2
        safety = agent.query_H(env)
        corrupt = env.episode_return - safety > 0
        agent.learn_C(corrupt)

        margin = env.episode_return - safety
        history['returns'].update(env.episode_return)
        history['safeties'].update(safety)
        history['margins'].update(margin)
        if margin > 0:
            history['margins_support'].update(margin)
        t += 1

    history['returns'].reset()
    history['safeties'].reset()
    history['margins'].reset()
    history['margins_support'].reset()

    return agent, env, args, history
