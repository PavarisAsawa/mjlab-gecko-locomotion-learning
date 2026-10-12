"""Gait phase clock (Tian et al. 2026, "Learning Whole-Body Control for a
Salamander Robot", Sec. III-B "Phase input").

The phase is only an input to the policy and to the foot-height reward; it does
not generate joint commands. Each foot j has phase phi_j in [-pi, pi):
  phi_j in [-pi, 0): stance, phi_j in [0, pi): swing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GaitPhaseCommand(CommandTerm):
  """Free-running clock; per-foot phases are fixed offsets from it.

  ``command`` is (sin(phi), cos(phi)) for every foot, shape (num_envs, 2 * num_feet).
  The clock starts at a random phase on reset so the policy sees all phases.
  """

  cfg: GaitPhaseCommandCfg

  def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._offsets = torch.tensor(cfg.phase_offsets, device=self.device)
    self._dphase = 2.0 * math.pi * cfg.frequency * env.step_dt
    self.clock = torch.zeros(self.num_envs, device=self.device)

  @property
  def phase(self) -> torch.Tensor:
    """Per-foot phase in [-pi, pi), shape (num_envs, num_feet)."""
    return wrap_to_pi(self.clock.unsqueeze(1) + self._offsets)

  @property
  def command(self) -> torch.Tensor:
    phase = self.phase
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.clock[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      -math.pi, math.pi
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:  # Per-step advance only; resets keep the sampled phase.
      self.clock = wrap_to_pi(self.clock + self._dphase)


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  frequency: float = 1.0
  """Gait frequency [Hz]."""
  phase_offsets: tuple[float, ...] = (0.0, math.pi, math.pi, 0.0)
  """Per-foot phase offsets [rad], in the foot order used by the rewards.
  Default: trot for (lf, rf, lh, rh); diagonal pairs move together."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  """Only resampled (randomized) on reset."""

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
