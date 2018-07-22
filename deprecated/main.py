from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from common import RandomAgent, SingleActionAgent, TabularQAgent, DeepQAgent, utils as ut
import ssrl.agents as ssrl

import argparse
import tqdm
import numpy as np

# TODO: Generalize to other gridworlds

parser = argparse.ArgumentParser(description='Learning (Hopefully) Safe Agents in Gridworlds')
parser.add_argument('--seed', type=int,
    help='Random seed')
parser.add_argument('-T', '--T', type=int, default=10000,
    help='Max timesteps')
parser.add_argument('--discount', type=float, default=.99,
    help='Agent-death probability complement. x_x')
parser.add_argument('-L', '--lr', type=float, default=.2,
    help='Learning rate')
parser.add_argument('-E', '--epsilon', type=float, default=.1,
    help='Exploration constant for epsilon greedy policy')
parser.add_argument('-R', '--decay-rate', type=float)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
    help='Which to device to use for DQN in PyTorch')
parser.add_argument('--replay-capacity', type=int, default=1000,
    help='Capacity of replay buffer')

# SSRL args
parser.add_argument('-S', '--ssrl', dest='ssrl', action='store_true')
parser.add_argument('-C', '--cheat', dest='cheat', action='store_true')
parser.set_defaults(ssrl=False, cheat=False)
parser.add_argument('-B', '--budget', type=int, default=200,
    help='Max number of queries to H for supervision in SSRL')
parser.add_argument('-W', '--budget-warmup', type=float, default=.5,
    help='Proportion of budget to spend during warmup phase')
parser.add_argument('-F', '--fuzzy-query', type=float, default=1.,
    help='Probability of querying H while online (only if buget and delta conditions are met)')
parser.add_argument('--delta', type=float, default=.9,
    help='Minimum quantile of episode returns for allowing to query H')
parser.add_argument('--C-prior', type=float, default=.01,
    help='Prior for state corruption')

args = parser.parse_args()
np.random.seed(args.seed)

returns = ut.AverageMeter(include_history=True)
safeties = ut.AverageMeter()
margins = ut.AverageMeter()
margins_support = ut.AverageMeter()

# Environment
env = TomatoWateringEnvironment()

# Agent
agent = ssrl.SSQAgent(env, args) if args.ssrl else TabularQAgent(env, args)

# Warmup phase
rando = RandomAgent(env, args) # Exploration only


if args.ssrl:
    print("#### WARMUP ####\n")
    warmup_phase = int(args.budget * args.budget_warmup)
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

else:
    print("#### SKIPPING WARMUP ####\n")


# Learning phase

print("\n#### LEARNING ####\n")
done = True
for t in tqdm.tqdm(range(args.T)):
    if done:
        (step_type, reward, discount, state), done = env.reset(), False

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if not args.cheat:
        agent.learn(state, action, reward, successor)
    else:
        agent.learn(state, action, env._get_hidden_reward(), successor)

    done = step_type.value == 2
    if done:
        # Query supervisor, if needed
        episode_return = env.episode_return
        try:
            delta_condition = episode_return > returns.quantile(args.delta)
        except IndexError:
            delta_condition = True
        fuzzy_condition = np.random.sample() < args.fuzzy_query
        if args.ssrl:
            if agent.budget > 0 and delta_condition and fuzzy_condition:
                safety = agent.query_H(env)
                safeties.update(safety)
                margins.update(env.episode_return - safety)
                corrupt = episode_return - safety > 0
                if corrupt:
                    # print('Episode return: {}'.format(env.episode_return))
                    # print('Safety performance: {}'.format(safety))
                    margins_support.update(env.episode_return - safety)
        else:
            safety = env.get_last_performance()
            safeties.update(safety)
            margins.update(env.episode_return - safety)
            corrupt = episode_return - safety > 0
            if corrupt:
                # print('Episode return: {}'.format(env.episode_return))
                # print('Safety performance: {}'.format(safety))
                margins_support.update(env.episode_return - safety)

        returns.update(env.episode_return)

        # Update corruption map
        if args.ssrl:
            agent.learn_C(corrupt)

    state = successor

print('Average episode return: {}'.format(returns.avg))
print('Average safety performance: {}'.format(safeties.avg))
print('Average margin: {} for {} corrupt reward histories in {} episodes'.format(margins_support.avg, margins_support.count, returns.count))
print('Average margin overall: {}'.format(margins.avg))
print('Overall performance: {}'.format(env.get_overall_performance()))
# print(len(agent.C_support), len(agent.C))
# print(agent.C_support)
