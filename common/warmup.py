import ssrl.warmup as ss

def dqn_warmup(agent, env, args):
    """Warm start for DQN agent"""
    returns = ut.AverageMeter(include_history=False)
    safeties = ut.AverageMeter()
    margins = ut.AverageMeter()
    margins_support = ut.AverageMeter()
    rando = RandomAgent(env, args) # Exploration only
    print("#### WARMUP ####\n")

    for i in range(args.replay_capacity):
        (step_type, reward, discount, state), done = env.reset(), False
        while not done:
            action = rando.act(None)
            step_type, reward, discount, successor = env.step(action)
            done = step_type.value == 2


def noop(agent, env, args):
    return agent, env, args, None

warmup_map = {
    'random':noop,
    'single':noop,
    'tabular-q':noop,
    'deep-q':dqn_warmup,
    'tabular-ssq':ss.random_warmup,
}