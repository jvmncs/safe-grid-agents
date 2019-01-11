"""Auto-constructs a CLI from relevant YAML config files."""
import sys

from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from ai_safety_gridworlds.environments.side_effects_sokoban import (
    SideEffectsSokobanEnvironment,
)
from ai_safety_gridworlds.environments.tomato_crmdp import TomatoCRMDPEnvironment


from safe_grid_agents.common.agents import (
    RandomAgent,
    SingleActionAgent,
    TabularQAgent,
    DeepQAgent,
    PPOMLPAgent,
    PPOCNNAgent,
    PPOCRMDPAgent,
)
from safe_grid_agents.ssrl import TabularSSQAgent
from safe_grid_agents.parsing import core_config, env_config, agent_config
import yaml
import argparse
import copy
from typing import Dict
from safe_grid_agents.types import EnvAlias, EnvName, Agent, AgentName

# Mapping of envs/agents to Python classes
env_map = {  # Dict[EnvAlias, EnvName]
    "bandit": "friend_foe",
    "belt": "conveyor_belt",
    "boat": "boat_race",
    "interrupt": "safe_interruptibility",
    "island": "island_navigation",
    "lava": "distributional_shift",
    "sokoban": "side_effects_sokoban",
    "super": "absent_supervisor",
    "tomato": "tomato_watering",
    "tomato-crmdp": "tomato_crmdp",
    "whisky": "whisky_gold",
}

agent_map = {  # Dict[AgentName, Agent]
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
type_map = {"float": float, "int": int, "str": str}  # Dict[str, type]


def map_type(x):
    try:
        return type_map[x]
    except KeyError:
        return x


def handle_parser_args(parsers, name, configs):
    """Assist adding arguments from `configs` to parser `name` from collection `parsers`."""
    p = parsers[name]
    config = configs[name]
    try:
        for key in list(config.keys()):
            argattrs = {k: map_type(v) for k, v in config.pop(key).items()}
            alias = argattrs.pop("alias")
            p.add_argument("-{}".format(alias), "--{}".format(key), **argattrs)
    except AttributeError:
        return


# Import yaml configs
with open(core_config) as core_yaml:
    core_parser_configs = yaml.load(core_yaml)
with open(env_config, "r") as env_yaml:
    env_parser_configs = yaml.load(env_yaml)
with open(agent_config, "r") as agent_yaml:
    agent_parser_configs = yaml.load(agent_yaml)
stashed_apcs = copy.deepcopy(agent_parser_configs)


def prepare_parser():
    """Create all CLI parsers/subparsers."""
    # Handle core parser args
    parser = argparse.ArgumentParser(
        description="Learning (Hopefully) Safe Agents in Gridworlds"
    )
    handle_parser_args({"core": parser}, "core", core_parser_configs)

    # Handle environment subparser args
    env_subparsers = parser.add_subparsers(
        help="Types of gridworld environments", dest="env_alias"
    )
    env_subparsers.required = True
    env_parsers = {}
    for env_name in env_map:
        env_parsers[env_name] = env_subparsers.add_parser(env_name)
        handle_parser_args(env_parsers, env_name, env_parser_configs)

    # Handle agent subparser args
    agent_subparsers = {}
    for env_name, env_parser in env_subparsers.choices.items():
        agent_parser_configs = copy.deepcopy(stashed_apcs)
        agent_subparsers[env_name] = env_parser.add_subparsers(
            help="Types of agents", dest="agent_alias"
        )
        agent_subparsers[env_name].required = True
        agent_parsers = {}
        for agent_name in agent_map:
            agent_parsers[agent_name] = agent_subparsers[env_name].add_parser(
                agent_name
            )
            handle_parser_args(agent_parsers, agent_name, agent_parser_configs)

    return parser
