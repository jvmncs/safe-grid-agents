"""Main safe-grid-agents script with CLI."""
from safe_grid_agents.parsing import prepare_parser, env_map, agent_map
from safe_grid_agents.common.warmup import warmup_map
from safe_grid_agents.common.learn import learn_map
from safe_grid_agents.common.eval import eval_map
from safe_grid_agents.common import utils as ut
from safe_grid_gym.envs import GridworldEnv

import safe_grid_gym

import gym
import os
import time
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    if args.disable_cuda:
        args.device = "cpu"

    # Create logging directory
    # The try/except is there in case log_dir is None,
    # in which case we use the TensorboardX default
    try:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
    except TypeError:
        args.log_dir = None

    # Get relevant env, agent, warmup function
    env_name = env_map[args.env_alias]
    agent_class = agent_map[args.agent_alias]
    warmup_fn = warmup_map[args.agent_alias]
    learn_fn = learn_map[args.agent_alias]
    eval_fn = eval_map[args.agent_alias]

    # Trackers
    history = ut.make_meters({})
    eval_history = ut.make_meters({})
    writer = SummaryWriter(args.log_dir)
    history["writer"] = writer
    eval_history["writer"] = writer

    # Instantiate, warmup
    env = gym.make(env_name)
    agent = agent_class(env, args)
    agent, env, history, args = warmup_fn(agent, env, history, args)

    # Learn and occasionally eval
    history["t"], eval_history["period"] = 0, 0
    init_state = env.reset()
    env_state = init_state, 0.0, False, {}
    for episode in range(args.episodes):
        history["episode"] = episode
        env_state, history, eval_next = learn_fn(agent, env, env_state, history, args)

        if eval_next:
            eval_history = eval_fn(agent, env, eval_history, args)
            eval_next = False

        env_state = env.reset(), 0.0, False, {}

    # One last eval
    eval_history = eval_fn(agent, env, eval_history, args)
