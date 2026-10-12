"""Observation helpers for the gecko locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["com_velocity_spine_frame", "foot_height"]


def com_velocity_spine_frame(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Whole-body CoM velocity (vx, vy, wz) in the spine heading frame.

  Following Tian et al. (Sec. III-C), velocities are taken at the equivalent
  centre of mass instead of the IMU, which oscillates with the body undulation.
  The heading frame is yaw-aligned with the line from the hind girdle to the
  front girdle, so it does not swing with the front body segment either.

  asset_cfg.body_names must be (front_girdle, hind_girdle) with preserve_order.
  Returns (num_envs, 3): CoM linear x/y velocity and mass-weighted yaw rate.
  """
  asset: Entity = env.scene[asset_cfg.name]
  mass = asset.data.model.body_mass[:, asset.indexing.body_ids]  # (W or 1, B)
  total = mass.sum(dim=1, keepdim=True)
  lin_w = (mass.unsqueeze(-1) * asset.data.body_com_lin_vel_w).sum(dim=1) / total
  yaw_rate = (mass * asset.data.body_com_ang_vel_w[..., 2]).sum(dim=1) / total[:, 0]

  girdles = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (N, 2, 3)
  axis = girdles[:, 0, :2] - girdles[:, 1, :2]
  heading = torch.atan2(axis[:, 1], axis[:, 0])
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  vx = cos_h * lin_w[:, 0] + sin_h * lin_w[:, 1]
  vy = -sin_h * lin_w[:, 0] + cos_h * lin_w[:, 1]
  return torch.stack([vx, vy, yaw_rate], dim=1)


def foot_height(
  env: ManagerBasedRlEnv, foot_rest_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Foot clearance above a flat floor at z=0, shape (num_envs, num_feet)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - foot_rest_height
