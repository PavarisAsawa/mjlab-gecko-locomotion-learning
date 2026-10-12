"""Spawn the gecko on a flat floor and open the MuJoCo viewer.

The scene is built with mjlab's Scene (same floor, actuators and physics options
as training), so what you see here is what the policy will train on. The motors
hold the default pose (ctrl = joint position targets).

Usage: uv run python tests/spawn_gecko.py
"""

import os

# See scripts/view_robot.py: GLFW's Wayland build aborts on GNOME 50.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mujoco  # noqa: E402

from mjlab.scene import Scene, SceneCfg  # noqa: E402
from mjlab.sim.sim import MujocoCfg  # noqa: E402
from mjlab.terrains.terrain_entity import TerrainEntityCfg  # noqa: E402
from mj_lab_gecko_project.robots.gecko import get_gecko_robot_cfg  # noqa: E402

# Physics options used by the locomotion task (mjlab velocity_env_cfg).
SIM_CFG = MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)


def make_gecko_on_floor() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """Compile robot + flat floor and return the model and data at the spawn pose."""
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_gecko_robot_cfg()},
    ),
    device="cpu",
  )
  model = scene.compile()
  SIM_CFG.apply(model)
  data = mujoco.MjData(model)
  # Hold the spawn pose: position-mode actuators track ctrl = current joint angle.
  for i in range(model.nu):
    data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
  mujoco.mj_forward(model, data)
  return model, data


if __name__ == "__main__":
  import mujoco.viewer

  model, data = make_gecko_on_floor()
  mujoco.viewer.launch(model, data)
