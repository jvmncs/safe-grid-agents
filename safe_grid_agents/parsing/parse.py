import yaml
import argparse
import copy
import sys

sys.path.append("ai-safety-gridworlds/")
from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from ai_safety_gridworlds.environments.side_effects_sokoban import (
    SideEffectsSokobanEnvironment
)
from safe_grid_agents.common.agents import (
    RandomAgent,
    SingleActionAgent,
    TabularQAgent,
    DeepQAgent,
)
from safe_grid_agents.ssrl import TabularSSQAgent
from . import core_config, env_config, agent_config
from typing import Dict
from ..types import Env, EnvName, Agent, AgentName

# Mapping of envs/agents to Python classes
env_map: Dict[EnvName, Env] = {
    # 'super':AbsentSupervisorEnvironment,
    "boat": BoatRaceEnvironment,
    # 'belt':ConveyorBeltEnvironment,
    # 'lava':DistributionalShiftEnvironment,
    # 'bandit':FriendFoeEnvironment,
    # 'island':IslandNavigationEnvironment,
    # 'interrupt':SafeInterruptibilityEnvironment,
    "sokoban": SideEffectsSokobanEnvironment,
    "tomato": TomatoWateringEnvironment,
    # 'whisky':WhiskyOrGoldEnvironment,
}
agent_map: Dict[AgentName, Agent] = {
    "random": RandomAgent,
    "single": SingleActionAgent,
    "tabular-q": TabularQAgent,
    "deep-q": DeepQAgent,
    "tabular-ssq": TabularSSQAgent,
}

# YAML conversion helper
type_map: Dict[str, type] = {"float": float, "int": int, "str": str}


def map_type(x):
    try:
        return type_map[x]
    except KeyError:
        return x


def handle_parser_args(parsers, name, configs):
    """Helper for adding arguments from `configs` to parser `name` from
    collection `parsers`"""
    p = parsers[name]
    config = configs[name]
    try:
        for key in list(config.keys()):
            argattrs = {k: map_type(v) for k, v in config.pop(key).items()}
            alias = argattrs.pop("alias")
            p.add_argument(f"-{alias}", f"--{key}", **argattrs)
    except AttributeError as e:
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
    """Takes care of all CLI parser/subparser creation."""
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
    for env_name in env_map.keys():
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
        for agent_name in agent_map.keys():
            agent_parsers[agent_name] = agent_subparsers[env_name].add_parser(
                agent_name
            )
            handle_parser_args(agent_parsers, agent_name, agent_parser_configs)

    return parser
