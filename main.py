from parsing import prepare_parser, env_map, agent_map
from common.warmup import warmup_map

if __name__=='__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    
    # Get relevant env, agent, warmup functiyon
    env_class = env_map[args.env_alias]
    agent_class = agent_map[args.agent_alias]
    warmup_fn = warmup_map[args.agent_alias]

    # Instantiate, warmup
    env = env_class()
    agent = agent_class(env, args)
    agent, env, args, state = warmup_fn(agent, env, args)

    returns = state['returns']
    # Learn

    # Eval
