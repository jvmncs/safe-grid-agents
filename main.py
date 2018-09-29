from safe_grid_agents.parsing import prepare_parser, env_map, agent_map
from safe_grid_agents.common.warmup import warmup_map
from safe_grid_agents.common.learn import learn_map
from safe_grid_agents.common.eval import eval_map
from safe_grid_agents.common import utils as ut

import os
import errno
import time
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    # Create logging directory
    try:
        if not os.path.exists(args.log_dir):
            try:
                os.makedirs(args.log_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    except TypeError:
        args.log_dir = None

    # Get relevant env, agent, warmup function
    env_class = env_map[args.env_alias]
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
    env = env_class()
    agent = agent_class(env, args)
    agent, env, history, args = warmup_fn(agent, env, history, args)

    # Learn and occasionally eval
    eval_next = False
    done = True
    episode = 0
    eval_history["period"] = 0
    for t in range(args.timesteps):
        if done:
            history = ut.track_metrics(episode, history, env)
            env_state, done = env.reset(), False
            episode += 1
            if eval_next:
                eval_history = eval_fn(agent, env, eval_history, args)
                eval_next = False
            time0 = time.time()
        env_state, history = learn_fn(t, agent, env, env_state, history, args)

        done = env_state[0].value == 2
        if t % args.eval_every == args.eval_every - 1:
            eval_next = True

    eval_history = eval_fn(agent, env, eval_history, args)
