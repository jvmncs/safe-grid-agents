safe-grid-agents
========
Training (hopefully) safe agents in gridworlds

## Layout
- `ai-safety-gridworlds`: Submodule of DeepMind's official [repo](https://github.com/deepmind/ai-safety-gridworlds).
- `safe_grid_agents/common`: Core codebase. Includes abstract base classes for a variety of agents, standard RL agents, their associated warmup functions, and a utilities file.
- `main.py`: Python executable for composing training jobs.
- `safe_grid_agents/parsing`: Helpers that construct a flexible CLI for `main.py`.
- `safe_grid_agents/ssrl`: Agents that implement semi-supervised reinforcement learning and their associated warmup functions.

## Adding agents
Steps to take when adding a new agent.

1. Determine where the agent should live; for example, if you're testing a new baseline from standard RL, include it in `common`, but if you're adding a new SSRL agent, add it to `ssrl`.  We'll refer to this folder as `<top>`.
2. (optional) If your agent doesn't fall into these categories, create a new top-level subdirectory `<top>` for it (using an informative abbreviation).  You should also create an abstract base class establishing the distinguishing functionality of your agent class in `<top>/base.py`. For example:
    - SSRL requires a stronger agent H to learn from, so we require a `query_H` method for each agent.
    - Additionally, following [Everitt et al.](https://arxiv.org/abs/1705.08417), we require a `learn_C` method to learn the probability of the state being corrupt.
3. Implement a warmup function in `<top>/warmup.py`, and make sure it's importable from `common/warmup.py`.
4. Add a new entry for the agent's CLI arguments in `parsing/agent_parser_configs.yaml`.  Follow the existing pattern and check for previously implemented anchors that cover the arguments you need (e.g. `epsilon-decay`).  These configs should be organized by where they appear in the folder structure of the repository.
5. TODO Add a function defining the agent-environment interaction.
6. TODO Add a function defining agent evaluation in the environment.
