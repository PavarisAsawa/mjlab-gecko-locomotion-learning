"""Open the gecko robot in the MuJoCo viewer to check the model.

Usage: uv run python scripts/view_robot.py
"""

import os

# GLFW's Wayland build loads GTK via libdecor, which aborts on GNOME 50
# ("xsettings ... does not contain a key named 'antialiasing'"). Use X11 instead.
# Must be set before glfw is imported.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mujoco.viewer  # noqa: E402

from mjlab.entity.entity import Entity  # noqa: E402
from mj_lab_gecko_project.robots.gecko import get_gecko_robot_cfg  # noqa: E402

if __name__ == "__main__":
  robot = Entity(get_gecko_robot_cfg())
  mujoco.viewer.launch(robot.spec.compile())
