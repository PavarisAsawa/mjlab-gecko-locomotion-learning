"""Gecko flat-ground locomotion environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from mj_lab_gecko_project.robots.gecko import (
  FOOT_GEOM_NAMES,
  FOOT_REST_HEIGHT,
  FOOT_SITE_NAMES,
  GECKO_ACTION_SCALE,
  SPINE_BODY_NAMES,
  get_gecko_robot_cfg,
)
from mj_lab_gecko_project.tasks.locomotion.locomotion_env_cfg import (
  make_locomotion_env_cfg,
)


def gecko_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gecko walking on flat ground (Tian et al. 2026, terrestrial, no backlash)."""
  return make_locomotion_env_cfg(
    robot_cfg=get_gecko_robot_cfg(),
    actuated_joints=r"joint\d_.*",
    action_scale=GECKO_ACTION_SCALE,
    foot_site_names=FOOT_SITE_NAMES,  # (lf, rf, lh, rh) = gait phase order.
    foot_geom_names=FOOT_GEOM_NAMES,
    girdle_body_names=("base_link", "body3"),  # Front girdle, hind girdle.
    spine_body_names=SPINE_BODY_NAMES,
    foot_rest_height=FOOT_REST_HEIGHT,
    play=play,
  )
