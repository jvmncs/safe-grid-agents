import os
import random
import subprocess

import ray
from ray import tune

from safe_grid_agents.parsing import prepare_parser
from train import train
from tune_config import TUNE_KWARGS, tune_config


parser = prepare_parser()
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randrange(500)

if args.disable_cuda:
    args.device = "cpu"

args.commit_id = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .decode("utf8")
    .strip()
)


######## Logging into TensorboardX ########
# If `args.log_dir` is None, we attempt a default unique up to env, agent, cheating, and seed combinations.
if args.log_dir is None:
    cheating = "baseline" if args.cheat else "corrupt"
    args.log_dir = os.path.join(
        "runs", args.env_alias, args.agent_alias, cheating, str(args.seed)
    )
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)


if args.tune is not None:
    ray.init()
    config = tune_config(args)

    # This lets us use argparse on top of ray tune while conforming
    # to Tune's requirement that the train function take exactly 2
    # arguments.
    tune.register_trainable(
        "train_curried_fn", lambda config, reporter: train(args, config, reporter)
    )

    # TODO(alok) Integrate Tune reporter with tensorboardX?
    experiment_spec = tune.Experiment(
        name="CRMDP", run="train_curried_fn", stop={}, config=config, **TUNE_KWARGS
    )

    tune.run_experiments(experiments=experiment_spec)
else:
    train(args)
