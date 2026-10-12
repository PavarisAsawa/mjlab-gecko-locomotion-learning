"""Residual joint position action with a first-order low-pass filter.

q_des = q_nominal + scale * a_t (residual action, Tian et al. Eq. 4-5), then
filtered before execution (Sec. III-E "Action Alignment"). The filter runs at
the policy rate so the same code can run on the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FilteredJointPositionAction(JointPositionAction):
  cfg: FilteredJointPositionActionCfg

  def __init__(self, cfg: FilteredJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    # Discrete first-order low-pass: y += alpha * (x - y).
    self._alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.cutoff_hz * env.step_dt)
    self._filtered = torch.zeros_like(self._raw_actions)
    self._filtered[:] = self._offset

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._filtered += self._alpha * (self._processed_actions - self._filtered)
    self._processed_actions = self._filtered.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    offset = self._offset
    self._filtered[env_ids] = offset[env_ids] if isinstance(offset, torch.Tensor) else offset


@dataclass(kw_only=True)
class FilteredJointPositionActionCfg(JointPositionActionCfg):
  cutoff_hz: float = 4.0
  """Low-pass cutoff frequency [Hz]. Use the same value on the robot."""

  def build(self, env: ManagerBasedRlEnv) -> FilteredJointPositionAction:
    return FilteredJointPositionAction(self, env)
