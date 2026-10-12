"""Headless evaluation of a trained gecko policy on fixed velocity commands.

For each command, runs N parallel robots for a few seconds (no pushes, no
observation noise) and reports the achieved whole-body CoM velocity, falls,
mechanical power and swing foot clearance.

Usage:
  uv run python scripts/evaluate_policy.py logs/rsl_rl/gecko_locomotion/<run>/model_1499.pt
"""

import argparse
import os
from dataclasses import asdict

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402

from mj_lab_gecko_project.robots.gecko import FOOT_REST_HEIGHT, FOOT_SITE_NAMES  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion import mdp  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion.config.gecko.env_cfgs import (  # noqa: E402
  gecko_flat_env_cfg,
)

TASK = "Gecko-Locomotion-Flat"
# (vx [m/s], vy [m/s], wz [rad/s])
COMMANDS = [
  (0.0, 0.0, 0.0),
  (0.1, 0.0, 0.0),
  (0.2, 0.0, 0.0),
  (0.3, 0.0, 0.0),
  (-0.1, 0.0, 0.0),
  (0.0, 0.1, 0.0),
  (0.0, 0.0, 0.4),
  (0.2, 0.0, 0.3),
]


def evaluate(checkpoint: str, num_envs: int, warmup_s: float, measure_s: float) -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"{'command (vx, vy, wz)':>24} | {'achieved CoM (vx, vy, wz)':>27} | falls | power W | swing clr cm")
  for cmd in COMMANDS:
    cfg = gecko_flat_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (cmd[0], cmd[0])
    twist.ranges.lin_vel_y = (cmd[1], cmd[1])
    twist.ranges.ang_vel_z = (cmd[2], cmd[2])
    twist.rel_forward_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    twist.debug_vis = False

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
    agent_cfg = load_rl_cfg(TASK)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    base = env.unwrapped
    girdles = SceneEntityCfg("robot", body_names=("base_link", "body3"), preserve_order=True)
    feet = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)
    joints = SceneEntityCfg("robot", joint_names=(r"joint\d_.*",))
    for c in (girdles, feet, joints):
      c.resolve(base.scene)

    obs, _ = env.reset()
    dt = base.step_dt
    vel, power, falls, n = torch.zeros(3, device=device), 0.0, 0, 0
    max_clear = torch.zeros(num_envs, len(FOOT_SITE_NAMES), device=device)
    with torch.inference_mode():
      for step in range(int((warmup_s + measure_s) / dt)):
        obs, _, dones, extras = env.step(policy(obs))
        if step * dt < warmup_s:
          continue
        vel += mdp.com_velocity_spine_frame(base, girdles).mean(0)
        power += mdp.energy_cost(base, joints).mean().item()
        max_clear = torch.maximum(max_clear, mdp.foot_height(base, FOOT_REST_HEIGHT, feet))
        falls += int((dones.bool() & ~extras["time_outs"].bool()).sum().item())
        n += 1
    vel /= n
    print(
      f"{str(cmd):>24} | {f'({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f})':>27} | "
      f"{falls:5d} | {power / n:7.2f} | {100 * max_clear.mean():8.1f}"
    )
    env.close()


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--warmup-s", type=float, default=2.0)
  ap.add_argument("--measure-s", type=float, default=6.0)
  a = ap.parse_args()
  evaluate(a.checkpoint, a.num_envs, a.warmup_s, a.measure_s)
