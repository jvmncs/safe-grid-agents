from safe_grid_agents.common.agents import RandomAgent


def random_warmup(agent, env, history, args):
    """Warm start for SSRL agent."""
    rando = RandomAgent(env, args)  # Exploration only
    print("#### WARMUP ####\n")
    warmup_phase = int(args.budget * args.warmup)
    t = 0
    # Random agent doesn't actually need states, rewards, or discounts to make
    # a decision, so we leave those out.
    while t < warmup_phase:
        (step_type, _, _, _), done = env.reset(), False
        while not done:
            action = rando.act(None)
            step_type, _, _, _ = env.step(action)
            done = step_type.value == 2
        safety = agent.query_H(env)
        corrupt = env.episode_return - safety > 0
        agent.learn_C(corrupt)

        margin = env.episode_return - safety
        history["returns"].update(env.episode_return)
        history["safeties"].update(safety)
        history["margins"].update(margin)
        if margin > 0:
            history["margins_support"].update(margin)
        t += 1

    history["returns"].reset()
    history["safeties"].reset()
    history["margins"].reset()
    history["margins_support"].reset()

    return agent, env, args, history
