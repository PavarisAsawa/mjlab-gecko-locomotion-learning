"""Spawn the gecko on a flat floor and check it stands still in the default pose."""

import math

import mujoco
import numpy as np
import pytest
from spawn_gecko import make_gecko_on_floor

SETTLE_S = 2.0
HOLD_S = 1.0
FEET = ("foot_lf", "foot_rf", "foot_lh", "foot_rh")


def _body(model, name: str) -> int:
  return model.body(f"robot/{name}").id


def _bodies_touching_floor(model, data) -> set[str]:
  floor = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith("terrain")}
  out = set()
  for c in data.contact[: data.ncon]:
    if c.geom1 in floor or c.geom2 in floor:
      other = c.geom2 if c.geom1 in floor else c.geom1
      out.add(model.body(model.geom_bodyid[other]).name.removeprefix("robot/"))
  return out


@pytest.fixture(scope="module")
def settled():
  model, data = make_gecko_on_floor()
  for _ in range(int(SETTLE_S / model.opt.timestep)):
    mujoco.mj_step(model, data)
  max_vel = 0.0
  z0 = data.xpos[_body(model, "base_link")][2]
  for _ in range(int(HOLD_S / model.opt.timestep)):
    mujoco.mj_step(model, data)
    max_vel = max(max_vel, float(np.abs(data.qvel).max()))
  return model, data, max_vel, z0


def test_model_structure():
  model, data = make_gecko_on_floor()
  assert model.nu == 19  # 3 spine + 4 legs x 4 joints
  assert data.ncon == 0, "robot should spawn above the floor, not in it"
  mass = model.body_subtreemass[_body(model, "base_link")]
  assert 2.0 < mass < 3.5, f"unexpected total mass {mass:.3f} kg"


def test_state_finite(settled):
  _, data, _, _ = settled
  assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_stands_upright(settled):
  model, data, _, z0 = settled
  base = _body(model, "base_link")
  up_z = data.xmat[base][8]  # world-z component of the base z-axis
  assert up_z > math.cos(math.radians(10)), f"base tilted: up_z={up_z:.3f}"
  z = data.xpos[base][2]
  assert 0.08 < z < 0.16, f"base height {z:.3f} m"
  assert abs(z - z0) < 2e-3, "base still sinking/rising after settling"


def test_at_rest(settled):
  _, _, max_vel, _ = settled
  assert max_vel < 0.05, f"robot jitters at rest: max |qvel| = {max_vel:.3f}"


def test_only_feet_touch_floor(settled):
  model, data, _, _ = settled
  touching = _bodies_touching_floor(model, data)
  assert touching == set(FEET), f"bodies touching floor: {sorted(touching)}"


def test_motors_within_limits(settled):
  model, data, _, _ = settled
  # Holding a 2.7 kg robot should need well under the 4.1 N*m stall torque.
  assert np.abs(data.actuator_force).max() < 2.0
