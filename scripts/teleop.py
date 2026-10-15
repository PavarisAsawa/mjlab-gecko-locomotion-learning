"""Drive the gecko with the keyboard using a trained policy.

The policy walks; you set its velocity command (vx, vy, wz) live, like the
gamepad in Tian et al. Uses the play config of Gecko-Locomotion-Flat (no
pushes, no observation noise, no timeout) with a single robot on the floor.

Usage:
  uv run python scripts/teleop.py                      # latest checkpoint in logs/
  uv run python scripts/teleop.py path/to/model_1499.pt

Keys (focus the viewer window; edit KEYMAP below to change them):
  Up / Down                    forward / backward   (vx +- 0.05 m/s)
  Left / Right                 turn left / right    (wz +- 0.1 rad/s)
  + / -  (also PgUp/PgDn, Home/End)  step left / right  (vy +- 0.05 m/s)
  Numpad 8/2, 4/6, 7/9         forward/back, step left/right, turn left/right
  Delete, Numpad 5             stop (zero command)
  Enter              reset robot          Space   pause
  [ / ]              slower / faster sim  R       toggle command arrows
Commands are clamped to the training ranges.
"""

import argparse
import os
import threading
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
# See scripts/view_robot.py: GLFW's Wayland build aborts on GNOME 50.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco  # noqa: E402
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.viewer import NativeMujocoViewer  # noqa: E402
from mjlab.viewer.native import keys as K  # noqa: E402

from mj_lab_gecko_project.tasks.locomotion.config.gecko.env_cfgs import (  # noqa: E402
  gecko_flat_env_cfg,
)

TASK = "Gecko-Locomotion-Flat"
LOG_ROOT = Path("logs/rsl_rl/gecko_locomotion")
STEP = {"vx": 0.05, "vy": 0.05, "wz": 0.1}

# Key -> (command, direction). Edit freely; several keys may map to one action.
# Avoid letter keys: MuJoCo's viewer uses every letter as a display toggle (J = joints,
# W = wireframe, ...), so they would also change the view. mjlab additionally uses
# Enter, Space, ",", ".", P, R, A and Right (single step while paused). mjlab's
# sim-speed keys - and = are taken over for vy here; sim speed moves to [ and ].
KEYMAP: dict[int, tuple[str, int]] = {
  # Arrows / PageUp-PageDown / Home-End.
  K.KEY_UP: ("vx", +1), K.KEY_DOWN: ("vx", -1),
  K.KEY_LEFT: ("wz", +1), K.KEY_RIGHT: ("wz", -1),
  K.KEY_PAGE_UP: ("vy", +1), K.KEY_PAGE_DOWN: ("vy", -1),
  K.KEY_HOME: ("vy", +1), K.KEY_END: ("vy", -1),
  # + / - : step left / right. "+" is the "=" key (with or without Shift).
  K.KEY_EQUAL: ("vy", +1), K.KEY_MINUS: ("vy", -1),
  K.KEY_KP_ADD: ("vy", +1), K.KEY_KP_SUBTRACT: ("vy", -1),
  # Numpad layout (Num Lock on): 8/2 forward/back, 4/6 left/right, 7/9 turn.
  K.KEY_KP_8: ("vx", +1), K.KEY_KP_2: ("vx", -1),
  K.KEY_KP_4: ("vy", +1), K.KEY_KP_6: ("vy", -1),
  K.KEY_KP_7: ("wz", +1), K.KEY_KP_9: ("wz", -1),
}
STOP_KEYS = (K.KEY_DELETE, K.KEY_KP_5)


class VelocityKeyboard:
  """Keyboard -> (vx, vy, wz). Called from the viewer thread, read by the main loop."""

  def __init__(self, ranges):
    self._limits = {"vx": ranges.lin_vel_x, "vy": ranges.lin_vel_y, "wz": ranges.ang_vel_z}
    self._cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    self._lock = threading.Lock()

  def __call__(self, key: int) -> None:
    with self._lock:
      if key in STOP_KEYS:
        self._cmd = dict.fromkeys(self._cmd, 0.0)
      elif key in KEYMAP:
        name, sign = KEYMAP[key]
        lo, hi = self._limits[name]
        value = round(self._cmd[name] + sign * STEP[name], 3)
        self._cmd[name] = min(max(value, lo), hi)

  def get(self) -> tuple[float, float, float]:
    with self._lock:
      return self._cmd["vx"], self._cmd["vy"], self._cmd["wz"]


class TeleopPolicy:
  """Writes the keyboard command into the env's twist command, then runs the policy.

  The command takes effect from the next observation (one 20 ms step of lag).
  """

  def __init__(self, policy, env: ManagerBasedRlEnv, keyboard: VelocityKeyboard):
    self._policy = policy
    self._twist = env.command_manager.get_term("twist")
    self._keyboard = keyboard

  def __call__(self, obs):
    self._twist.vel_command_b[:] = torch.tensor(self._keyboard.get(), device=self._twist.device)
    return self._policy(obs)


class TeleopViewer(NativeMujocoViewer):
  """Native viewer with the current command shown in the top-left overlay."""

  def __init__(self, *args, keyboard: VelocityKeyboard, **kwargs):
    super().__init__(*args, key_callback=keyboard, **kwargs)
    self._keyboard = keyboard

  def _safe_key_callback(self, key: int) -> None:
    if key in (K.KEY_MINUS, K.KEY_EQUAL):  # vy, not mjlab's sim-speed change.
      self._keyboard(key)
    elif key == K.KEY_LEFT_BRACKET:
      self.request_speed_down()
    elif key == K.KEY_RIGHT_BRACKET:
      self.request_speed_up()
    else:
      super()._safe_key_callback(key)

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    vx, vy, wz = self._keyboard.get()
    status = self.get_status()
    viewer.set_texts((
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      "vx  (Up/Down)\nvy  (+/-)\nwz  (Left/Right)\n\nStatus\nActual RT",
      f"{vx:+.2f} m/s\n{vy:+.2f} m/s\n{wz:+.2f} rad/s\n\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}\n{status.actual_realtime:.2f}x",
    ))


def latest_checkpoint() -> Path:
  runs = sorted(p for p in LOG_ROOT.glob("*/") if any(p.glob("model_*.pt")))
  if not runs:
    raise SystemExit(f"No checkpoints under {LOG_ROOT}/. Train first or pass a path.")
  return max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("checkpoint", nargs="?", type=Path, help="model_*.pt (default: latest)")
  args = ap.parse_args()
  checkpoint = args.checkpoint or latest_checkpoint()
  print(f"[teleop] checkpoint: {checkpoint}")

  # Keyboard limits = training command ranges.
  keyboard = VelocityKeyboard(gecko_flat_env_cfg().commands["twist"].ranges)

  cfg = gecko_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  twist = cfg.commands["twist"]
  # The keyboard owns the command: no random resampling, standing or forward envs.
  twist.resampling_time_range = (1e9, 1e9)
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 0.0

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
  runner = MjlabOnPolicyRunner(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = TeleopPolicy(runner.get_inference_policy(device=device), env.unwrapped, keyboard)

  print(__doc__[__doc__.index("Keys"):])
  TeleopViewer(env, policy, keyboard=keyboard).run()
  env.close()


if __name__ == "__main__":
  main()
