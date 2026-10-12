"""Reward terms from Tian et al. 2026 (Table I).

  r_v      = exp(-||v_cmd,xy - v_b,xy||^2 / sigma_v)
  r_w      = exp(-(w_cmd,z - w_b,z)^2 / sigma_w)
  r_energy = -sum_i |qdot_i * tau_i|          (here: positive cost, negative weight)
  r_phase  = exp(-sum_j (z_j^foot - z_j^ref(phi_j))^2 / sigma_phase)

v_b and w_b are whole-body CoM velocities (see observations.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import GaitPhaseCommand
from .observations import com_velocity_spine_frame, foot_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "energy_cost",
  "feet_phase_consistency",
  "swing_height_reference",
  "track_com_ang_vel",
  "track_com_lin_vel",
]


def track_com_lin_vel(
  env: ManagerBasedRlEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  vel = com_velocity_spine_frame(env, asset_cfg)
  error = torch.sum(torch.square(command[:, :2] - vel[:, :2]), dim=1)
  return torch.exp(-error / sigma)


def track_com_ang_vel(
  env: ManagerBasedRlEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  vel = com_velocity_spine_frame(env, asset_cfg)
  return torch.exp(-torch.square(command[:, 2] - vel[:, 2]) / sigma)


def energy_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Absolute mechanical power sum_i |qdot_i * tau_i| over the actuated joints."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(tau * qd), dim=1)


def swing_height_reference(phase: torch.Tensor, swing_height: float) -> torch.Tensor:
  """Bezier foot-height profile: 0 in stance (phase < 0), smooth rise and fall
  to ``swing_height`` in swing (phase in [0, pi)), peaking at mid-swing.

  Each half of the swing is a cubic Bezier with control points (0, 0, 1, 1),
  i.e. B(s) = 3 s^2 - 2 s^3 (zero slope at lift-off, apex and touch-down).
  """
  t = torch.clamp(phase / torch.pi, 0.0, 1.0)  # Swing progress; 0 in stance.
  s = torch.where(t < 0.5, 2.0 * t, 2.0 - 2.0 * t)
  return swing_height * (3.0 * s**2 - 2.0 * s**3)


def feet_phase_consistency(
  env: ManagerBasedRlEnv,
  sigma: float,
  swing_height: float,
  foot_rest_height: float,
  gait_command_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track the phase-conditioned foot height. When the velocity command is
  (near) zero the reference is all feet on the ground, so the robot stands."""
  gait = env.command_manager.get_term(gait_command_name)
  assert isinstance(gait, GaitPhaseCommand)
  ref = swing_height_reference(gait.phase, swing_height)
  command = env.command_manager.get_command(command_name)
  moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
  ref = ref * moving.unsqueeze(1)
  height = foot_height(env, foot_rest_height, asset_cfg)
  error = torch.sum(torch.square(height - ref), dim=1)
  return torch.exp(-error / sigma)
