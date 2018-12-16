"""Custom types for safe-grid-agents."""
# TODO put Transitions in namedtuple/dataclass
from typing import Dict, Tuple, NamedTuple
import torch
import numpy as np
import collections

Transition = object
History = Dict[str, object]

AgentName = str
Agent = object

EnvAlias = str
EnvName = str

Rollout = collections.namedtuple("Rollout", ["states", "actions", "rewards", "returns"])
Experience = NamedTuple(
    "Experience",
    [
        ("state", np.ndarray),
        ("action", int),
        ("reward", float),
        ("successor", np.ndarray),
        ("terminal", bool),
    ],
)

ExperienceBatch = NamedTuple(
    "ExperienceBatch",
    [
        ("states", torch.Tensor),
        ("actions", torch.Tensor),
        ("rewards", torch.Tensor),
        ("successors", torch.Tensor),
        ("terminals", torch.Tensor),
    ],
)
