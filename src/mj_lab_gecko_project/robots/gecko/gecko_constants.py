"""Gecko robot constants.

The MJCF in xmls/ is generated from the USD by scripts/usd_to_mjcf.py; re-run it
after changing the USD instead of editing gecko.xml by hand.

Joints (all actuated by Dynamixel XM430-W350 @ 12 V):
  spine:  joint0_f, joint0_m, joint0_h
  legs:   joint{1..4}_{lf,rf,lh,rh}
Passive: foot_{leg}_x / foot_{leg}_y (+-15 deg ankle compliance, unactuated).
"""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GECKO_XML: Path = Path(__file__).parent / "xmls" / "gecko.xml"
assert GECKO_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GECKO_XML))


##
# Names.
##

LEGS = ("lf", "rf", "lh", "rh")
FOOT_SITE_NAMES = tuple(f"foot_{leg}" for leg in LEGS)  # Sites at the foot origins.
FOOT_GEOM_NAMES = tuple(f"foot_{leg}_collision" for leg in LEGS)
SPINE_BODY_NAMES = ("base_link", "body1", "body2", "body3")  # Front -> hind girdle.
SPINE_JOINTS = r"joint0_.*"
LEG_JOINTS = r"joint[1-4]_.*"
# Height of the foot site above the ground when the pad rests flat (cylinder
# half-height + offset, see scripts/usd_to_mjcf.py).
FOOT_REST_HEIGHT = 0.0075

##
# Actuator config.
##

# Datasheet values at the output shaft (ROBOTIS e-Manual, XM430-W350 @ 12 V).
SUPPLY_VOLTAGE = 12.0
XM430_W350 = DcMotorDatasheetParams(
  nominal_voltage=12.0,
  stall_torque=4.1,  # N*m
  no_load_speed=46.0 * 2.0 * math.pi / 60.0,  # rad/s
)

# Firmware position P gain (register "Position P Gain", default 800). The X-series
# controller computes PWM = (P / 128) * error[ticks], with 885 = 100% duty, so in
# volts per radian of error:
DXL_POSITION_P_GAIN = 800
KP_VOLT_PER_RAD = (
  (DXL_POSITION_P_GAIN / 128) * (4096 / (2.0 * math.pi)) * (SUPPLY_VOLTAGE / 885)
)

GECKO_ACTUATOR_CFG = BuiltinDcMotorActuatorCfg(
  target_names_expr=(r"joint\d_.*",),
  motor_params=XM430_W350,
  mode=DcMotorInputMode.POSITION,
  stiffness=KP_VOLT_PER_RAD,
  damping=0.0,  # Position D gain = 0; back-EMF provides the damping.
  voltage_limit=SUPPLY_VOLTAGE,
  # TODO: identify on hardware (see system-ID notes). Placeholder values.
  armature=0.01,
  frictionloss=0.05,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  # Root height from the converter: lowest collision point + 5 mm at zero pose.
  pos=(0.0, 0.0, 0.1407),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Feet win the friction/condim merge against the floor (priority=1), so foot
# friction domain randomization actually takes effect.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,  # Belly/legs keep friction too, so the robot cannot slide on them.
  priority={r"foot_.*_collision": 1, r".*": 0},
  friction={r"foot_.*_collision": (0.8,)},
)

##
# Final config.
##

GECKO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GECKO_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_gecko_robot_cfg() -> EntityCfg:
  """Get a fresh gecko robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GECKO_ARTICULATION,
  )


# Policy action (unit-scale) -> joint position offset [rad].
# Spine range is +-15 deg, legs +-45 deg / 110 deg, so the spine gets a smaller scale.
GECKO_ACTION_SCALE: dict[str, float] = {SPINE_JOINTS: 0.25, LEG_JOINTS: 0.5}
