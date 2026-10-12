"""Flat-ground whole-body locomotion task.

Follows Tian et al. 2026, "Learning Whole-Body Control for a Salamander Robot"
(arXiv:2603.16683), terrestrial part only, without backlash modelling:

  - Action: residual joint offsets on a nominal pose, low-pass filtered,
    tracked by the servos at 50 Hz policy rate (Sec. III-A, III-E).
  - Observation: joint vel, joint offset, IMU angular velocity, projected gravity,
    gait phase, velocity command, last 3 actions; no linear velocity (Sec. III-B).
  - Reward: CoM linear/yaw velocity tracking, energy, phase-conditioned foot
    height (Table I), with w_v=1.0, w_w=0.5, w_energy=1e-3, w_phase=1.0. mjlab
    multiplies every weight by dt=0.02, as in the paper.
  - Domain randomization: pushes, friction, mass distribution, joint friction,
    PD gains (Sec. III-E).

Robot-specific names are passed in by config/<robot>/env_cfgs.py.
"""

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mj_lab_gecko_project.tasks.locomotion import mdp


def make_locomotion_env_cfg(
  *,
  robot_cfg: EntityCfg,
  actuated_joints: str,
  action_scale: dict[str, float],
  foot_site_names: tuple[str, ...],
  foot_geom_names: tuple[str, ...],
  girdle_body_names: tuple[str, str],
  spine_body_names: tuple[str, ...],
  foot_rest_height: float,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat-ground locomotion config.

  Args:
    robot_cfg: Entity config of the robot.
    actuated_joints: Regex matching the actuated joints (excludes passive ones,
      which have no encoder on hardware and must not be observed).
    action_scale: Residual action scale [rad] per joint regex.
    foot_site_names: Foot sites, in the order of the gait phase offsets.
    foot_geom_names: Foot collision geoms (friction randomization).
    girdle_body_names: (front_girdle, hind_girdle); defines the heading frame.
    spine_body_names: Bodies whose COM is randomized.
    foot_rest_height: Foot site height above ground when standing.
    play: Evaluation settings (no noise, pushes or episode timeouts).
  """
  robot = SceneEntityCfg("robot")
  joints = SceneEntityCfg("robot", joint_names=(actuated_joints,))
  girdles = SceneEntityCfg("robot", body_names=girdle_body_names, preserve_order=True)
  feet = SceneEntityCfg("robot", site_names=foot_site_names, preserve_order=True)

  ##
  # Observations (Sec. III-B, Eq. 6).
  ##

  actor_terms = {
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": joints},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "joint_pos": ObservationTermCfg(  # Delta q = q - q_nominal.
      func=mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": joints},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "gait_phase": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "gait"}
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "actions": ObservationTermCfg(  # {a_t-1, a_t-2, a_t-3}.
      func=mdp.last_action, history_length=3
    ),
  }

  # Asymmetric critic: noise-free actor obs plus privileged state.
  critic_terms = {
    **{k: ObservationTermCfg(func=v.func, params=v.params, history_length=v.history_length)
       for k, v in actor_terms.items()},
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": joints}),
    "com_vel": ObservationTermCfg(
      func=mdp.com_velocity_spine_frame, params={"asset_cfg": girdles}
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"foot_rest_height": foot_rest_height, "asset_cfg": feet},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions (Sec. III-A): q_des = q_nominal + scale * a, low-pass filtered.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.FilteredJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(actuated_joints,),
      scale=action_scale,
      use_default_offset=True,  # q_nominal = robot_cfg.init_state.joint_pos.
      cutoff_hz=4.0,
    )
  }

  ##
  # Commands.
  ##

  commands: dict[str, CommandTermCfg] = {
    # Ranges from the paper's hardware tests: vx 0.3 m/s, vy 0.15 m/s, wz 0.45 rad/s.
    "twist": mdp.UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      rel_standing_envs=0.1,
      rel_forward_envs=0.2,
      heading_command=False,
      debug_vis=True,
      ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.15, 0.3),
        lin_vel_y=(-0.15, 0.15),
        ang_vel_z=(-0.5, 0.5),
      ),
    ),
    "gait": mdp.GaitPhaseCommandCfg(
      frequency=1.0,
      phase_offsets=(0.0, math.pi, math.pi, 0.0),  # Trot, order = foot_site_names.
    ),
  }

  ##
  # Events: resets and domain randomization (Sec. III-E).
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.01), "yaw": (-math.pi, math.pi)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0), "asset_cfg": joints},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.4, 0.4)}},
    ),
    "foot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
    "mass_distribution": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={"alpha_range": (-0.05, 0.05), "asset_cfg": robot},  # mass x e^(2 alpha): +-10%.
    ),
    "spine_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=spine_body_names),
        "operation": "add",
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.005, 0.005)},
      },
    ),
    "joint_friction": EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={"asset_cfg": joints, "operation": "abs", "ranges": (0.02, 0.1)},
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={"asset_cfg": robot, "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2)},
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": joints, "bias_range": (-0.01, 0.01)},
    ),
  }

  ##
  # Rewards (Table I).
  ##

  rewards = {
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_com_lin_vel,
      weight=1.0,
      params={"sigma": 0.05, "command_name": "twist", "asset_cfg": girdles},
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_com_ang_vel,
      weight=0.5,
      params={"sigma": 0.1, "command_name": "twist", "asset_cfg": girdles},
    ),
    "energy": RewardTermCfg(
      func=mdp.energy_cost, weight=-1.0e-3, params={"asset_cfg": joints}
    ),
    "feet_phase": RewardTermCfg(
      func=mdp.feet_phase_consistency,
      weight=1.0,
      params={
        "sigma": 0.01,
        "swing_height": 0.03,
        "foot_rest_height": foot_rest_height,
        "gait_command_name": "gait",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": feet,
      },
    ),
    # Not in the paper: light safety regularizers. Set weight 0 to disable.
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": joints}
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    ),
  }

  if play:
    events.pop("push_robot")
    commands["twist"].rel_standing_envs = 0.0

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      num_envs=1,
      extent=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=girdle_body_names[0],
      distance=1.2,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,  # 0.005 s x 4 = 50 Hz policy (paper: 50 Hz).
    episode_length_s=int(1e9) if play else 20.0,  # 1000 steps (Table II).
  )
