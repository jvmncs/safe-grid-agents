"""Main safe-grid-agents script with CLI."""
import random

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

import safe_grid_gym
from safe_grid_agents.common import utils as ut
from safe_grid_agents.common.eval import EVAL_MAP
from safe_grid_agents.common.learn import LEARN_MAP
from safe_grid_agents.common.warmup import WARMUP_MAP
from safe_grid_agents.parsing import AGENT_MAP, ENV_MAP


def noop(*args, **kwargs):
    pass


def train(args, config=None, reporter=noop):
    # TODO(alok) This is here because there were issues with registering custom
    # environments in each run. This should be looked at and removed.
    import safe_grid_gym

    # Use Ray Tune's `config` arguments where appropriate by merging.
    if config is not None:
        vars(args).update(config)

    # fix seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get relevant env, agent, warmup function
    env_name = ENV_MAP[args.env_alias]
    agent_class = AGENT_MAP[args.agent_alias]
    warmup_fn = WARMUP_MAP[args.agent_alias]
    learn_fn = LEARN_MAP[args.agent_alias]
    eval_fn = EVAL_MAP[args.agent_alias]

    history, eval_history = ut.make_meters({}), ut.make_meters({})

    writer = SummaryWriter(args.log_dir)
    for k, v in args.__dict__.items():
        writer.add_text("data/{}".format(k), str(v))

    history["writer"] = writer
    eval_history["writer"] = writer

    env = gym.make(env_name)
    env.seed(args.seed)

    agent = agent_class(env, args)

    agent, env, history, args = warmup_fn(agent, env, history, args)

    ######## Learn (and occasionally evaluate) ########
    history["t"], eval_history["period"] = 0, 0

    for episode in range(args.episodes):
        env_state = (
            env.reset(),
            0.0,
            False,
            {"hidden_reward": 0.0, "observed_reward": 0.0},
        )
        history["episode"] = episode
        env_state, history, eval_next = learn_fn(agent, env, env_state, history, args)
        info = env_state[3]
        reporter(
            hidden_reward=info["hidden_reward"], obs_reward=info["observed_reward"]
        )

        if eval_next:
            eval_history = eval_fn(agent, env, eval_history, args)
            eval_next = False

    # One last evaluation.
    eval_history = eval_fn(agent, env, eval_history, args)
