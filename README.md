# safe-grid-agents

Training (hopefully) safe agents in gridworlds.

Emphasizing extensibility, modularity, and accessibility.

## Layout

-   `safe_grid_agents/common`: Core codebase. Includes abstract base
    classes for a variety of agents, their associated warmup/learn/eval
    functions, and a utilities file.
-   `main.py`: Python executable for composing training jobs.
-   `safe_grid_agents/parsing`: Helpers that construct a flexible CLI
    for `main.py`.
-   `safe_grid_agents/ssrl`: Agents that implement semi-supervised
    reinforcement learning and their associated warmup functions.

## Installation

When installing with pip, make sure to use the
`process-dependency-links` flag:

``` {.sh}
pip install . --process-dependency-links
```

URL-based dependencies are available for audit at the following
repositories and forks: -
[safe-grid-gym](https://github.com/jvmancuso/safe-grid-gym) -
[ai-safety-gridworlds](https://github.com/jvmancuso/ai-safety-gridworlds)

If you plan on developing this library, make sure to add an `-e` flag to
the above pip install command.

This repo requires [tensorboardX](https://github.com/lanpa/tensorboardX)
for monitoring and visualizing agent learning, as well as PyTorch for
implementation of certain agents. Currently, tensorboardX does not
function properly without Tensorflow installed. Since the installation
process of these packages can vary system to system, we exclude them
from our build process. There are multiple tutorials online for
installing both of these online. For example, on OS X without CUDA
support I'd go with:

``` {.sh}
# Replace `tensorflow` with `tensorflow-gpu` if you have a GPU.
pip install torch torchvision tensorflow
```

# Usage

## Training agents

You can use the CLI to `main.py` to modularly drop agents into arbitrary
safety gridworlds. For example, `python main.py boat tabular-q --lr .5`
will train a TabularQAgent on the BoatRaceEnvironment with a learning
rate of 0.5.

There are a number of customizable parameters to modify training runs.
These parameters are split into three groups: - Core arguments: args
that are shared across all agents/environments. Found in
[`parsing/core_parser_configs.yaml`](https://github.com/jvmancuso/safe-grid-agents/blob/master/safe_grid_agents/parsing/core_parser_configs.yaml).
- Environment arguments: args specific to environments but shared across
agents. Currently empty, but could be useful for specific environments,
depending on the agent. Found in
[`parsing/env_parser_configs.yaml`](https://github.com/jvmancuso/safe-grid-agents/blob/master/safe_grid_agents/parsing/env_parser_configs.yaml).
- Agent environments: args specific to agents. Most hyperparameters live
here. Found in
[`parsing/agent_parser_configs.yaml`](https://github.com/jvmancuso/safe-grid-agents/blob/master/safe_grid_agents/parsing/agent_parser_configs.yaml).

The generalized form for the CLI is

``` {.sh}
python main.py <core_args> env <env_args> agent <agent_args>
```

## Ray Tune

We support using Ray Tune to configure hyperparameters. Look at
`TUNE_DEFAULT_CONFIG` in `main.py` to see which are currently supported.
If you specify a tunable parameter on the CLI with the `-t` or `--tune`
flag, it will be automatically set.

### Example

This will automatically set parameters for the learning rate `lr` and
discount rate `discount`.

``` {.sh}
# `-t` and `--tune` are equivalent, and can be used interchangeably.
python3 main.py -t lr --tune discount boat tabular-q
```

## Monitoring agent learning with tensorboardX

You can use the `--log-dir`/`-L` flag to the main.py script to specify a
directory for saving training and evaluation metrics across runs. I
suggest a pattern similar to

``` {.sh}
logs/sokoban/deep-q/lr5e-4
# that is, <logdir>/<env_alias>/<agent_alias>/<uniqueid_or_hparams>
```

If no log-dir is specified for main.py, logging defaults to the `runs/`
directory, which can be helpful to separate debugging runs from training
runs.

Given a log directory `<logs>`, simply run `tensorboard --logdir <logs>`
to visualize an agent's learning.

# Development

## Code style

We use [black](https://github.com/ambv/black) for auto-formatting
according to a consistent style guide. To auto format, run `black .`
from inside the repo folder. To make this more convenient, you can
install plugins for your preferred text editor that auto-format on every
save.

## Adding agents

Steps to take when adding a new agent.

1.  Determine where the agent should live; for example, if you're
    testing a new baseline from standard RL, include it in `common`, but
    if you're adding a new SSRL agent, add it to `ssrl`. We'll refer to
    this folder as `<top>`.
2.  (optional) If your agent doesn't fall into these categories, create
    a new top-level subdirectory `<top>` for it (using an informative
    abbreviation). You should also create an abstract base class
    establishing the distinguishing functionality of your agent class in
    `<top>/base.py`. For example:
    -   SSRL requires a stronger agent H to learn from, so we require a
        `query_H` method for each agent.
    -   Additionally, following [Everitt et
        al.](https://arxiv.org/abs/1705.08417), we require a `learn_C`
        method to learn the probability of the state being corrupt.
3.  (optional) Implement a warmup function in `<top>/warmup.py`, and
    make sure it's importable from `common/warmup.py`. The `noop`
    default warmup function works for agents that don't require any
    special functionality.
4.  Implement a function describing the agent's learning feedback loop
    in `<top>/learn.py`. See
    [`common/learn.py`](https://github.com/jvmancuso/safe-grid-agents/blob/master/safe_grid_agents/common/learn.py)
    for an example distinguishing DQN from a tabular Q-learning agent.
5.  (optional) Implement a function in `<top>/eval.py` describing the
    evaluation feedback loop. The `default_eval` function in
    `common/eval.py` should cover most cases, so you may not need to add
    anything for evaluation.
6.  Add a new entry for the agent's CLI arguments in
    `parsing/agent_parser_configs.yaml`. Follow the existing pattern and
    check for previously implemented YAML anchors that cover the
    arguments you need (e.g. `learnrate`, `epsilon-anneal`, etc.). These
    configs should be organized by where they appear in the folder
    structure of the repository.
