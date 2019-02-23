"""Auto-constructs a CLI from relevant YAML config files."""

from argparse import ArgumentParser
from copy import deepcopy

import yaml

from safe_grid_agents.common.agents import (
    DeepQAgent,
    PPOCNNAgent,
    PPOMLPAgent,
    RandomAgent,
    SingleActionAgent,
    TabularQAgent,
)
from safe_grid_agents.parsing import agent_config, core_config, env_config
from safe_grid_agents.spiky.agents import PPOCRMDPAgent
from safe_grid_agents.ssrl.agents import TabularSSQAgent


# Mapping of envs/agents to Python classes
ENV_MAP = {  # Dict[EnvAlias, EnvName]
    "bandit": "FriendFoe-v0",
    "belt": "ConveyorBelt-v0",
    "boat": "BoatRace-v0",
    "interrupt": "SafeInterruptibility-v0",
    "island": "IslandNavigation-v0",
    "lava": "DistributionalShift-v0",
    "sokoban": "SideEffectsSokoban-v0",
    "super": "AbsentSupervisor-v0",
    "tomato": "TomatoWatering-v0",
    "tomato-crmdp": "TomatoCrmdp-v0",
    "whisky": "WhiskyGold-v0",
    "corners": "ToyGridworldCorners-v0",
    "way": "ToyGridworldOnTheWay-v0",
    "trans-boat": "TransitionBoatRace-v0",
}

AGENT_MAP = {  # Dict[AgentName, Agent]
    "random": RandomAgent,
    "single": SingleActionAgent,
    "tabular-q": TabularQAgent,
    "deep-q": DeepQAgent,
    "ppo-mlp": PPOMLPAgent,
    "ppo-cnn": PPOCNNAgent,
    "ppo-crmdp": PPOCRMDPAgent,
    "tabular-ssq": TabularSSQAgent,
}

# YAML conversion helper
TYPE_MAP = {"float": float, "int": int, "str": str}  # Dict[str, type]


def map_type(x: str) -> type:
    try:
        return TYPE_MAP[x]
    except KeyError:
        return x


def handle_parser_args(parsers, name, configs) -> None:
    """Assist adding arguments from `configs` to parser `name` from collection
    `parsers`."""
    p, config = parsers[name], configs[name]

    try:
        # `list` is necessary since we pop the original config keys and we
        # can't use `deepcopy` as `dict.keys()` is a generator and is therefore
        # unpickleable.
        for key in list(config.keys()):
            argattrs = {k: map_type(v) for k, v in config.pop(key).items()}
            if "alias" in argattrs:
                alias = argattrs.pop("alias")
                p.add_argument("-{}".format(alias), "--{}".format(key), **argattrs)
            else:  # some arguments have no short form
                p.add_argument("--{}".format(key), **argattrs)
    except AttributeError:
        return


# Import yaml configs
with open(core_config) as core_yaml:
    core_parser_configs = yaml.load(core_yaml)
with open(env_config, "r") as env_yaml:
    env_parser_configs = yaml.load(env_yaml)
with open(agent_config, "r") as agent_yaml:
    agent_parser_configs = yaml.load(agent_yaml)
stashed_agent_parser_configs = deepcopy(agent_parser_configs)


def prepare_parser() -> ArgumentParser:
    """Create all CLI parsers/subparsers."""
    # Handle core parser args
    parser = ArgumentParser(
        description="Learning (Hopefully) Safe Agents in Gridworlds"
    )
    handle_parser_args({"core": parser}, "core", core_parser_configs)

    # Handle environment subparser args
    env_subparsers = parser.add_subparsers(
        help="Types of gridworld environments", dest="env_alias"
    )
    env_subparsers.required = True
    env_parsers = {}
    for env_name in ENV_MAP:
        env_parsers[env_name] = env_subparsers.add_parser(env_name)
        handle_parser_args(env_parsers, env_name, env_parser_configs)

    # Handle agent subparser args
    agent_subparsers = {}
    for env_name, env_parser in env_subparsers.choices.items():
        agent_parser_configs = deepcopy(stashed_agent_parser_configs)
        agent_subparsers[env_name] = env_parser.add_subparsers(
            help="Types of agents", dest="agent_alias"
        )
        agent_subparsers[env_name].required = True
        agent_parsers = {}
        for agent_name in AGENT_MAP:
            agent_parsers[agent_name] = agent_subparsers[env_name].add_parser(
                agent_name
            )
            handle_parser_args(agent_parsers, agent_name, agent_parser_configs)

    return parser
