"""MDP terms for the gecko locomotion task.

Re-exports mjlab's generic terms (mjlab.envs.mdp) plus the custom terms defined
in this package.
"""

from mjlab.envs.mdp import *  # noqa: F403
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommandCfg as UniformVelocityCommandCfg,
)

from .actions import FilteredJointPositionActionCfg as FilteredJointPositionActionCfg
from .commands import GaitPhaseCommand as GaitPhaseCommand
from .commands import GaitPhaseCommandCfg as GaitPhaseCommandCfg
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
