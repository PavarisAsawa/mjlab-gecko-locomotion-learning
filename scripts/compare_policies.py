"""Compare the four network/optimizer cases on the same Gecko-Locomotion-Flat env.

  1. MLP + PPO        logs/rsl_rl/gecko_locomotion/<run>/model_*.pt
  2. MLP + OpenES     logs/es/gecko_locomotion_es_ff/<run>/es_*.pt
  3. Hebbian + OpenES logs/es/gecko_locomotion_es_hebb/<run>/es_*.pt
  4. Hebbian + PPO    logs/rsl_rl/gecko_locomotion_hebb_ppo/<run>/model_*.pt

Every policy is evaluated with the same protocol (deterministic actions, the ES
mean parameters mu):
  A. Fixed commands (play env: no pushes, no obs noise): achieved CoM velocity,
     tracking error, falls, power, cost of transport, swing clearance, target rate.
  B. Training distribution (train env: random commands, pushes, noise, DR):
     task return per second over 20 s episodes and fall rate.
plus learning curves (Episode_Reward/* vs env steps) and training cost.

Usage:
  uv run python scripts/compare_policies.py                       # latest run of each case
  uv run python scripts/compare_policies.py --case "MLP+ES=logs/es/.../es_499.pt"
Results go to logs/compare/<date>/ (results.md, results.json, curves.png).
"""

import argparse
import json
import math
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402

import mj_lab_gecko_project.tasks  # noqa: E402, F401  (registers the tasks)
from mj_lab_gecko_project.robots.gecko import FOOT_REST_HEIGHT, FOOT_SITE_NAMES  # noqa: E402
from mj_lab_gecko_project.rl.es_models import RunningObsNorm, build_es_model  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion import mdp  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion.config.gecko.env_cfgs import (  # noqa: E402
  gecko_flat_env_cfg,
)

DEVICE = "cuda:0"
CASES = {  # name -> (log dir, checkpoint glob)
  "MLP+PPO": ("logs/rsl_rl/gecko_locomotion", "model_*.pt"),
  "MLP+ES": ("logs/es/gecko_locomotion_es_ff", "es_*.pt"),
  "Hebbian+ES": ("logs/es/gecko_locomotion_es_hebb", "es_*.pt"),
  "Hebbian+PPO": ("logs/rsl_rl/gecko_locomotion_hebb_ppo", "model_*.pt"),
}
PPO_TASKS = {"gecko_locomotion": "Gecko-Locomotion-Flat",
             "gecko_locomotion_hebb_ppo": "Gecko-Locomotion-Flat-HebbPPO"}
COMMANDS = [  # (vx, vy, wz)
  (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (-0.1, 0.0, 0.0),
  (0.0, 0.1, 0.0), (0.0, 0.0, 0.4), (0.0, 0.0, -0.4), (0.2, 0.0, 0.3), (0.1, 0.1, 0.2),
]


##
# Policies.
##


def _ckpt_index(p: Path) -> int:
  return int(p.stem.split("_")[1])


def latest_checkpoint(log_dir: str, pattern: str) -> Path | None:
  runs = sorted(p for p in Path(log_dir).glob("*/") if any(p.glob(pattern)))
  return max(runs[-1].glob(pattern), key=_ckpt_index) if runs else None


class Policy:
  """Uniform interface: act(obs TensorDict) -> actions, reset(done mask or None)."""

  def __init__(self, ckpt: Path, env: RslRlVecEnvWrapper) -> None:
    self.ckpt = ckpt
    self.is_es = ckpt.name.startswith("es_")
    n = env.num_envs
    if self.is_es:
      data = torch.load(ckpt, map_location="cpu", weights_only=False)
      sizes = data["sizes"]
      self.model = build_es_model(data["model"], n, sizes[0], sizes[-1], data["es_cfg"])
      self.model.set_params(np.tile(data["mu"], (n, 1)))
      self.norm = RunningObsNorm(sizes[0], DEVICE)
      self.norm.load_state_dict(data["obs_norm"])
      self.num_params = data["num_params"]
    else:
      task = PPO_TASKS[ckpt.parent.parent.name]
      runner = MjlabOnPolicyRunner(env, asdict(load_rl_cfg(task)), device=DEVICE)
      runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
      self.actor = runner.alg.actor
      self.actor.eval()
      self.num_params = sum(p.numel() for p in self.actor.parameters())

  def reset(self, done: torch.Tensor | None = None) -> None:
    if self.is_es:
      self.model.reset(None if done is None else done.nonzero().flatten())
    elif self.actor.is_recurrent:
      if done is None:
        self.actor.reset(None, None)
      else:
        self.actor.reset(done.long())

  def act(self, obs) -> torch.Tensor:
    if self.is_es:
      return self.model(self.norm(obs["actor"]))
    return self.actor(obs)


##
# Evaluation.
##


def make_env(play: bool, num_envs: int) -> RslRlVecEnvWrapper:
  cfg = gecko_flat_env_cfg(play=play)
  cfg.scene.num_envs = num_envs
  cfg.seed = 0  # Same terrain friction / mass draws and commands for every policy.
  if play:
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = twist.rel_forward_envs = 0.0
    twist.debug_vis = False
  return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEVICE))


@torch.inference_mode()
def eval_fixed_commands(ckpt: Path, num_envs: int, warmup_s: float, measure_s: float) -> dict:
  env = make_env(play=True, num_envs=num_envs)
  base = env.unwrapped
  policy = Policy(ckpt, env)
  girdles = SceneEntityCfg("robot", body_names=("base_link", "body3"), preserve_order=True)
  feet = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)
  joints = SceneEntityCfg("robot", joint_names=(r"joint\d_.*",))
  for c in (girdles, feet, joints):
    c.resolve(base.scene)
  robot = base.scene["robot"]
  mass = float(base.sim.mj_model.body_subtreemass[int(robot.indexing.root_body_id)])
  twist = base.command_manager.get_term("twist")
  action_term = base.action_manager.get_term("joint_pos")
  dt = base.step_dt

  rows = []
  for cmd in COMMANDS:
    cmd_t = torch.tensor(cmd, device=DEVICE)
    obs, _ = env.reset()
    policy.reset(None)
    twist.vel_command_b[:] = cmd_t
    obs = env.get_observations()
    vel = torch.zeros(3, device=DEVICE)
    power = rate = 0.0
    falls, n = 0, 0
    clear = torch.zeros(num_envs, len(FOOT_SITE_NAMES), device=DEVICE)
    prev_target = action_term._processed_actions.clone()
    for step in range(int((warmup_s + measure_s) / dt)):
      twist.vel_command_b[:] = cmd_t
      obs, _, dones, extras = env.step(policy.act(obs))
      done = dones.bool()
      policy.reset(done) if done.any() else None
      target = action_term._processed_actions
      if step * dt >= warmup_s:
        vel += mdp.com_velocity_spine_frame(base, girdles).mean(0)
        power += mdp.energy_cost(base, joints).mean().item()
        rate += ((target - prev_target).abs().mean() / dt).item()
        clear = torch.maximum(clear, mdp.foot_height(base, FOOT_REST_HEIGHT, feet))
        falls += int((done & ~extras["time_outs"].bool()).sum().item())
        n += 1
      prev_target = target.clone()
    v = (vel / n).tolist()
    speed = math.hypot(v[0], v[1])
    rows.append({
      "command": cmd, "achieved": v,
      "lin_err": math.hypot(cmd[0] - v[0], cmd[1] - v[1]), "yaw_err": abs(cmd[2] - v[2]),
      "falls": falls, "power_w": power / n,
      "cot": (power / n) / (mass * 9.81 * speed) if math.hypot(*cmd[:2]) > 0 and speed > 0.02 else None,
      "clearance_cm": 100 * clear.mean().item(), "target_rate": rate / n,
    })
  num_params = policy.num_params
  env.close()
  return {"rows": rows, "num_params": num_params}


@torch.inference_mode()
def eval_train_distribution(ckpt: Path, num_envs: int, episode_s: float) -> dict:
  env = make_env(play=False, num_envs=num_envs)
  policy = Policy(ckpt, env)
  obs, _ = env.reset()
  policy.reset(None)
  alive = torch.ones(num_envs, dtype=torch.bool, device=DEVICE)
  ret = torch.zeros(num_envs, device=DEVICE)
  for _ in range(int(episode_s / env.unwrapped.step_dt)):
    obs, rew, dones, extras = env.step(policy.act(obs))
    ret += rew * alive
    done = dones.bool()
    alive &= ~(done & ~extras["time_outs"].bool())
    policy.reset(done) if done.any() else None
  out = {"return_per_s": ret.mean().item() / episode_s, "fall_rate": 1.0 - alive.float().mean().item()}
  env.close()
  return out


##
# Training curves.
##


def _yaml_int(path: Path, key: str) -> int:
  """First `key: <int>` in an mjlab params yaml. These files contain !!python tags
  (tuples, enums), so they are not parsed as YAML."""
  return int(re.search(rf"^\s*{key}:\s*(\d+)", path.read_text(), re.M).group(1))


def read_curves(ckpt: Path) -> dict:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  run = ckpt.parent
  ea = EventAccumulator(str(next(run.glob("events*"))), size_guidance={"scalars": 0})
  ea.Reload()
  tags = [t for t in ea.Tags()["scalars"] if t.startswith("Episode_Reward/")]
  if "Perf/env_steps" in ea.Tags()["scalars"]:  # ES
    steps = {e.step: e.value for e in ea.Scalars("Perf/env_steps")}
    to_steps = lambda it: steps[it]  # noqa: E731
  else:  # PPO
    per_it = _yaml_int(run / "params/env.yaml", "num_envs") * _yaml_int(
      run / "params/agent.yaml", "num_steps_per_env"
    )
    to_steps = lambda it: (it + 1) * per_it  # noqa: E731
  curves = {}
  for t in tags:
    ev = ea.Scalars(t)
    curves[t.split("/", 1)[1]] = ([to_steps(e.step) for e in ev], [e.value for e in ev])
  first = ea.Scalars(tags[0])
  return {
    "curves": curves,
    "env_steps": curves[next(iter(curves))][0][-1],
    "wall_clock_h": (first[-1].wall_time - first[0].wall_time) / 3600,
  }


def plot_curves(results: dict, path: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  panels = ["total", "track_lin_vel", "track_ang_vel", "feet_phase", "energy"]
  fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4))
  for name, r in results.items():
    curves = r["training"]["curves"]
    xs = curves["track_lin_vel"][0]
    for ax, key in zip(axes, panels):
      if key == "total":
        ys = np.sum([np.interp(xs, *curves[k]) for k in curves], axis=0)
      else:
        ys = curves[key][1]
      ax.plot(np.array(xs) / 1e6, ys, label=name, lw=1.4)
  for ax, key in zip(axes, panels):
    ax.set_title(f"{key} (reward/s)")
    ax.set_xlabel("env steps (M)")
    ax.grid(alpha=0.3)
  axes[0].legend()
  fig.tight_layout()
  fig.savefig(path, dpi=130)


##
# Report.
##


def write_report(results: dict, out: Path) -> str:
  lines = ["# Gecko locomotion: network x optimizer comparison", ""]
  lines += ["## Summary", "",
            "| Case | Params | Env steps | Train time | Return/s (train dist.) | Fall rate | "
            "Lin. err (m/s) | Yaw err (rad/s) | Falls (fixed cmds) | Power (W) | CoT | Target rate (rad/s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
  for name, r in results.items():
    rows = r["fixed"]["rows"]
    cots = [x["cot"] for x in rows if x["cot"] is not None]
    lines.append(
      f"| {name} | {r['fixed']['num_params']:,} | {r['training']['env_steps'] / 1e6:.0f} M | "
      f"{r['training']['wall_clock_h']:.2f} h | {r['train_dist']['return_per_s']:.3f} | "
      f"{100 * r['train_dist']['fall_rate']:.1f}% | {np.mean([x['lin_err'] for x in rows]):.3f} | "
      f"{np.mean([x['yaw_err'] for x in rows]):.3f} | {sum(x['falls'] for x in rows)} | "
      f"{np.mean([x['power_w'] for x in rows]):.2f} | {np.mean(cots) if cots else float('nan'):.2f} | "
      f"{np.mean([x['target_rate'] for x in rows]):.2f} |"
    )
  lines += ["", "## Per command: achieved CoM velocity (vx, vy, wz)", "",
            "| Command | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
  for i, cmd in enumerate(COMMANDS):
    cells = []
    for r in results.values():
      x = r["fixed"]["rows"][i]
      a = x["achieved"]
      cells.append(f"({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})" + (f" ✗{x['falls']}" if x["falls"] else ""))
    lines.append(f"| {cmd} | " + " | ".join(cells) + " |")
  lines += ["", "Checkpoints: " + ", ".join(f"{k}: `{v['checkpoint']}`" for k, v in results.items()), ""]
  text = "\n".join(lines)
  (out / "results.md").write_text(text)
  return text


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--case", action="append", default=[], help="NAME=path/to/checkpoint (overrides discovery)")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--warmup-s", type=float, default=2.0)
  ap.add_argument("--measure-s", type=float, default=8.0)
  ap.add_argument("--episode-s", type=float, default=20.0)
  args = ap.parse_args()

  ckpts = {name: latest_checkpoint(*spec) for name, spec in CASES.items()}
  for item in args.case:
    name, path = item.split("=", 1)
    ckpts[name] = Path(path)
  out = Path("logs/compare") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out.mkdir(parents=True)

  results = {}
  for name, ckpt in ckpts.items():
    if ckpt is None:
      print(f"[compare] {name}: no checkpoint found, skipped")
      continue
    print(f"[compare] {name}: {ckpt}")
    training = read_curves(ckpt)  # Cheap; fail before the slow evaluations.
    results[name] = {
      "checkpoint": str(ckpt),
      "training": training,
      "fixed": eval_fixed_commands(ckpt, args.num_envs, args.warmup_s, args.measure_s),
      "train_dist": eval_train_distribution(ckpt, args.num_envs, args.episode_s),
    }
  (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
  plot_curves(results, out / "curves.png")
  print(write_report(results, out))
  print(f"\n[compare] wrote {out}/results.md, results.json, curves.png")


if __name__ == "__main__":
  main()
