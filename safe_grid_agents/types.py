"""Custom types for safe-grid-agents."""
# TODO put Transitions in namedtuple/dataclass
from typing import Dict, Tuple
import torch

Transition = object
History = Dict[str, object]

AgentName = str
Agent = object

EnvName = str
Env = object

Experiences = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
