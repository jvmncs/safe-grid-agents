import sys
sys.path.append('../')
from common.agents import RandomAgent
import common.utils as ut

def random_warmup(agent, env, args, verbose=True):
    """Warm start for SSRL agent"""
    returns = ut.AverageMeter(include_history=True)
    safeties = ut.AverageMeter()
    margins = ut.AverageMeter()
    margins_support = ut.AverageMeter()
    rando = RandomAgent(env, args) # Exploration only
    print("#### WARMUP ####\n")
    warmup_phase = int(args.budget * args.warmup)
    init_budget = args.budget
    for e in range(warmup_phase):
        (step_type, reward, discount, state), done = env.reset(), False
        while not done:
            action = rando.act(None)
            step_type, reward, discount, successor = env.step(action)
            done = step_type.value == 2
        safety = agent.query_H(env)
        corrupt = env.episode_return - safety > 0
        agent.learn_C(corrupt)

        margin = env.episode_return - safety
        returns.update(env.episode_return)
        safeties.update(safety)
        margins.update(margin)
        if margin > 0:
            margins_support.update(margin)

    print("### WARMUP STATS ###")
    print('Return:   {} avg | {} max | {} count'.format(returns.avg, returns.max, returns.count))
    print('Safety:   {} avg | {} max | {} count'.format(safeties.avg, safeties.max, safeties.count))
    print('Margin:   {} avg | {} max | {} count'.format(margins.avg, margins.max, margins.count))
    print('MSupport: {} avg | {} max | {} count'.format(
        margins_support.avg, margins_support.max, margins_support.count))

    returns.reset()
    safeties.reset()
    margins.reset()
    margins_support.reset()

    return agent, env, args, {"returns":returns}
