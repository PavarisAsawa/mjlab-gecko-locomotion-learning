"""Record each comparison case walking the same command sequence, then tile all
four into one side-by-side video.

Every case runs in the play env (no pushes, no observation noise, same seed),
one robot, a camera tracking the front girdle, and the same timed commands.
Each frame is labelled with the case, the command, the measured whole-body CoM
velocity and the distance travelled. The arrows drawn by mjlab show the
commanded (twist) and actual velocity.

Usage:
  uv run python scripts/record_videos.py                 # latest checkpoint of each case
  uv run python scripts/record_videos.py --out docs/comparison
Writes <out>/<case>.mp4 and <out>/comparison.mp4.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MUJOCO_GL", "egl")  # Headless offscreen rendering.

import mediapy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from compare_policies import CASES, DEVICE, Policy, latest_checkpoint  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion import mdp  # noqa: E402
from mj_lab_gecko_project.tasks.locomotion.config.gecko.env_cfgs import (  # noqa: E402
  gecko_flat_env_cfg,
)

# (duration s, (vx, vy, wz), label)
SEQUENCE = [
  (2.0, (0.0, 0.0, 0.0), "stand"),
  (5.0, (0.2, 0.0, 0.0), "forward 0.2 m/s"),
  (4.0, (0.3, 0.0, 0.0), "forward 0.3 m/s"),
  (4.0, (0.0, 0.0, 0.4), "turn left 0.4 rad/s"),
  (4.0, (0.2, 0.0, 0.3), "forward + turn"),
  (3.0, (0.0, 0.1, 0.0), "sideways 0.1 m/s"),
  (3.0, (-0.1, 0.0, 0.0), "backward 0.1 m/s"),
  (2.0, (0.0, 0.0, 0.0), "stop"),
]
WIDTH, HEIGHT = 640, 480


def _font(size: int) -> ImageFont.ImageFont:
  for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    if Path(path).exists():
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


FONT_BIG, FONT = _font(22), _font(16)


def label(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img, "RGBA")
  draw.rectangle([0, 0, WIDTH, 34 + 20 * len(lines)], fill=(0, 0, 0, 150))
  draw.text((10, 6), title, font=FONT_BIG, fill=(255, 255, 255, 255))
  for i, line in enumerate(lines):
    draw.text((10, 34 + 20 * i), line, font=FONT, fill=(230, 230, 230, 255))
  return np.asarray(img)


@torch.inference_mode()
def record(name: str, ckpt: Path) -> list[np.ndarray]:
  cfg = gecko_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.seed = 0
  cfg.viewer.width, cfg.viewer.height = WIDTH, HEIGHT
  twist_cfg = cfg.commands["twist"]
  twist_cfg.resampling_time_range = (1e9, 1e9)
  twist_cfg.rel_standing_envs = twist_cfg.rel_forward_envs = 0.0
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode="rgb_array"))
  base = env.unwrapped
  policy = Policy(ckpt, env)
  girdles = SceneEntityCfg("robot", body_names=("base_link", "body3"), preserve_order=True)
  girdles.resolve(base.scene)
  twist = base.command_manager.get_term("twist")
  robot = base.scene["robot"]
  dt = base.step_dt

  obs, _ = env.reset()
  policy.reset(None)
  start = robot.data.root_link_pos_w[0, :2].clone()
  frames, t = [], 0.0
  for duration, cmd, text in SEQUENCE:
    cmd_t = torch.tensor(cmd, device=DEVICE)
    for _ in range(int(round(duration / dt))):
      twist.vel_command_b[:] = cmd_t
      obs, _, dones, _ = env.step(policy.act(obs))
      if dones.bool().any():
        policy.reset(dones.bool())
      v = mdp.com_velocity_spine_frame(base, girdles)[0].tolist()
      dist = torch.linalg.norm(robot.data.root_link_pos_w[0, :2] - start).item()
      t += dt
      frames.append(label(base.render(), name, [
        f"t={t:5.1f}s  cmd: {text}",
        f"cmd    vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
        f"actual vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {v[2]:+.2f}",
        f"distance from start {dist:.2f} m",
      ]))
  env.close()
  return frames


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", type=Path, default=Path("docs/comparison"))
  ap.add_argument("--case", action="append", default=[], help="NAME=path/to/checkpoint")
  args = ap.parse_args()
  ckpts = {name: latest_checkpoint(*spec) for name, spec in CASES.items()}
  for item in args.case:
    name, path = item.split("=", 1)
    ckpts[name] = Path(path)
  ckpts = {k: v for k, v in ckpts.items() if v is not None}
  args.out.mkdir(parents=True, exist_ok=True)
  fps = 50  # One frame per policy step (50 Hz).

  videos = {}
  for i, (name, ckpt) in enumerate(ckpts.items(), start=1):
    print(f"[video] {name}: {ckpt}", flush=True)
    frames = record(f"{i}. {name}", ckpt)
    path = args.out / f"{name.replace('+', '_').lower()}.mp4"
    mediapy.write_video(path, frames, fps=fps)
    videos[name] = frames
    print(f"[video] wrote {path} ({len(frames)} frames)", flush=True)

  # 2x2 grid (pad with black tiles if fewer than 4 cases).
  tiles = list(videos.values())
  n = len(tiles[0])
  black = [np.zeros_like(tiles[0][0])] * n
  while len(tiles) < 4:
    tiles.append(black)
  grid = [
    np.concatenate([np.concatenate([tiles[0][k], tiles[1][k]], axis=1),
                    np.concatenate([tiles[2][k], tiles[3][k]], axis=1)], axis=0)
    for k in range(n)
  ]
  mediapy.write_video(args.out / "comparison.mp4", grid, fps=fps)
  print(f"[video] wrote {args.out / 'comparison.mp4'}", flush=True)


if __name__ == "__main__":
  main()
