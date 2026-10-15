# Gecko Locomotion Task (`Gecko-Locomotion-Flat`)

How the gecko robot learns to walk on flat ground in [mjlab](https://github.com/mujocolab/mjlab):
the robot model, actuators, observations, rewards, how to train and evaluate,
and what to watch out for.

The task follows the terrestrial part of Tian et al. 2026,
*Learning Whole-Body Control for a Salamander Robot*
([arXiv:2603.16683](https://arxiv.org/abs/2603.16683)). It differs from the paper in three ways:

- **Ground:** flat floor only, with no rough terrain and no swimming.
- **Backlash:** none. The paper adds passive joints at the girdles to model it.
- **Robot:** our gecko instead of their salamander. It has no tail and a shorter spine.

---

## 1. The gecko robot

| | |
|---|---|
| Size (L × W × H) | 0.71 × 0.42 × 0.18 m in the default pose |
| Total mass | 2.65 kg (spine 1.67 kg, each leg 0.25 kg) |
| Actuated joints | 19 Dynamixel XM430-W350 servos |
| Passive joints | 8 (two per ankle) |
| Source model | `robots/gecko/gecko-aug-2026.usd` (Isaac Sim export) |

### Kinematic structure

```
base_link (front girdle, root, IMU) ── joint1..4_lf ── foot_lf
   │                                 └ joint1..4_rf ── foot_rf
 joint0_f
 body1
 joint0_m
 body2
 joint0_h
 body3 (hind girdle) ─────────────── joint1..4_lh ── foot_lh
                                    └ joint1..4_rh ── foot_rh
```

| Joint | Role | Range |
|---|---|---|
| `joint0_f/m/h` | Spine yaw (3 segments) | ±15° |
| `joint1_*` | Leg joint 1 | ±45° |
| `joint2_*` | Leg joint 2 | −100°…10° for `lh`, `rf`; −10°…100° for `rh`, `lf` (mirrored) |
| `joint3_*` | Leg joint 3 | ±45° |
| `joint4_*` | Leg joint 4 | same as `joint2_*` |
| `foot_*_x`, `foot_*_y` | Passive ankle, 2 axes, **not actuated or observed** | ±15° |

Legs: `lf` left front, `rf` right front, `lh` left hind, `rh` right hind.

### From USD to MuJoCo

MuJoCo cannot read USD, so [`scripts/usd_to_mjcf.py`](../scripts/usd_to_mjcf.py) converts it into
`robots/gecko/xmls/gecko.xml` plus STL meshes. **Re-run it whenever the USD changes**, and
never edit `gecko.xml` by hand:

```bash
uv run --no-project --python 3.12 --with usd-core --with numpy python scripts/usd_to_mjcf.py \
  src/mj_lab_gecko_project/robots/gecko/gecko-aug-2026.usd \
  src/mj_lab_gecko_project/robots/gecko/xmls
```

The converter carries over the bodies, masses and inertias, joint frames and limits, and the
visual and collision meshes. It also makes three changes:

- **Foot collision is a cylinder** (r = 2 cm, 1.2 cm thick) instead of the CAD pad mesh.
  The mesh made the passive ankles chatter at about 1.3 rad/s while standing still; the
  cylinder brings that down to 0.002 rad/s.
- **Collision hulls are capped at 64 vertices.** Some CAD hulls had 3,000, which made GPU
  collision slow.
- **Feet get mass from their volume** (about 15 g each), because the USD doesn't author it.
  PhysX computes it the same way.

---

## 2. Setup

```bash
uv sync                          # creates .venv with Python 3.13 (mjlab needs < 3.14)
export GIT_PYTHON_REFRESH=quiet  # needed while git is not installed
export PYGLFW_LIBRARY_VARIANT=x11  # MuJoCo viewer crashes on GNOME 50 Wayland otherwise
```

Put both `export` lines in `~/.bashrc`.

Check that everything works:

```bash
uv run pytest tests                    # 9 tests: tasks load, robot spawns and stands still
uv run python tests/spawn_gecko.py     # viewer: robot standing on the floor
uv run list-envs                       # should list Gecko-Locomotion-Flat
```

### Where things live

```
src/mj_lab_gecko_project/
├── robots/gecko/
│   ├── gecko-aug-2026.usd       # source model
│   ├── xmls/                    # generated MJCF + meshes (do not edit)
│   └── gecko_constants.py       # actuators, init pose, collisions, names
└── tasks/locomotion/
    ├── locomotion_env_cfg.py    # the task: obs, actions, rewards, DR, sim settings
    ├── mdp/
    │   ├── actions.py           # residual action + low-pass filter
    │   ├── commands.py          # gait phase clock
    │   ├── observations.py      # CoM velocity, foot height
    │   └── rewards.py           # paper reward terms
    └── config/gecko/
        ├── env_cfgs.py          # plugs gecko names into the task
        ├── rl_cfg.py            # PPO settings
        └── __init__.py          # registers Gecko-Locomotion-Flat
scripts/
├── usd_to_mjcf.py               # USD → MJCF converter
├── evaluate_policy.py           # headless command-tracking evaluation
└── view_robot.py                # view the bare robot model
tests/                           # spawn_gecko.py, test_gecko_spawn.py, test_tasks.py
```

---

## 3. Actuator model (Dynamixel XM430-W350)

Defined in [`gecko_constants.py`](../src/mj_lab_gecko_project/robots/gecko/gecko_constants.py).

Every actuated joint uses `BuiltinDcMotorActuatorCfg` in POSITION mode, which is MuJoCo's native
`<dcmotor>`. This mirrors what the servo's own firmware does:

```
real:  goal position → firmware P controller → PWM × supply voltage → DC motor (back-EMF) → gearbox
sim:   target        → PID (kp in V/rad)     → voltage, clamped at 12 V → <dcmotor> torque–speed model
```

A plain PD controller misses two effects that matter here. Torque drops as speed rises, and
back-EMF adds strong built-in damping (about 0.85 N·m·s/rad at the output shaft).

| Parameter | Value | Source |
|---|---|---|
| Supply voltage | 12 V | — |
| Stall torque | 4.1 N·m | datasheet; matches the USD drive `maxForce` |
| No-load speed | 46 rpm (4.82 rad/s) | datasheet |
| Position P gain | 800 (firmware register) → **kp ≈ 55 V/rad** (≈ 19 N·m/rad near stall) | `(P/128)·(4096/2π)·(V/885)` |
| D gain | 0 | firmware default; back-EMF supplies the damping |
| `armature` | 0.01 kg·m² | ⚠ **placeholder, needs system ID** |
| `frictionloss` | 0.05 N·m | ⚠ **placeholder, needs system ID** |

**Settings to use on the real servos so they match the sim:**

- **Position P gain:** 800, the same value the sim assumes. If you change it, change
  `DXL_POSITION_P_GAIN` too.
- **Profiles off:** `Profile Velocity = 0` and `Profile Acceleration = 0`. Otherwise the
  firmware smooths goals in a way the sim doesn't model.
- **USB latency:** `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`
  (the default is 16 ms). Use SyncRead/SyncWrite at a high baud rate.

---

## 4. Task design

All settings are in
[`locomotion_env_cfg.py`](../src/mj_lab_gecko_project/tasks/locomotion/locomotion_env_cfg.py).

### Simulation

| | |
|---|---|
| Physics step | 5 ms (implicitfast integrator) |
| Policy rate | **50 Hz** (decimation 4), as in the paper |
| Episode | 20 s = 1,000 policy steps |
| Terrain | flat plane |
| Termination | timeout, or base tilted > 60° |

### Action space (19)

The policy outputs a **residual** around a nominal pose (paper Eq. 4–5):

```
q_des = q_nominal + scale · a_t          then low-pass filtered (4 Hz, first order)
```

| | |
|---|---|
| `q_nominal` | all joints 0 rad, the CAD pose, which stands stably (`INIT_STATE` in `gecko_constants.py`) |
| `scale` | spine 0.25 rad, legs 0.5 rad per unit action |
| Filter | `y += α(x − y)`, α = 1 − exp(−2π·4 Hz·0.02 s) ≈ 0.40, run at policy rate |

**Joint order** (actions, `joint_pos` and `joint_vel` observations all use it):

```
joint0_f, joint0_m, joint0_h,
joint1_lh, joint2_lh, joint3_lh, joint4_lh,
joint1_rh, joint2_rh, joint3_rh, joint4_rh,
joint1_lf, joint2_lf, joint3_lf, joint4_lf,
joint1_rf, joint2_rf, joint3_rf, joint4_rf
```

### Observation space

**Actor (112 dims).** It uses only what the real robot can measure. As in the paper, there is no
linear velocity.

| Term | Dims | Content | Noise (uniform ±) |
|---|---|---|---|
| `joint_vel` | 19 | q̇ | 0.5 rad/s |
| `joint_pos` | 19 | Δq = q − q_nominal | 0.01 rad |
| `base_ang_vel` | 3 | gyro, `base_link` frame | 0.2 rad/s |
| `projected_gravity` | 3 | gravity direction, `base_link` frame | 0.05 |
| `gait_phase` | 8 | sin φ for (lf, rf, lh, rh), then cos φ for (lf, rf, lh, rh) | — |
| `command` | 3 | (vx, vy, ωz) command | — |
| `actions` | 57 | last 3 actions, oldest first | — |

**Critic (119 dims).** The same terms without noise, plus privileged information:

| Extra term | Dims | Content |
|---|---|---|
| `com_vel` | 3 | whole-body CoM (vx, vy, ωz) in the spine frame |
| `foot_height` | 4 | foot clearance above the floor |

### Commands

**Velocity command `twist`**, resampled every 4–8 s:

| | Range |
|---|---|
| vx | −0.15 … 0.30 m/s |
| vy | −0.15 … 0.15 m/s |
| ωz | −0.5 … 0.5 rad/s |

- 10% of envs get a zero command (stand still).
- 20% get forward-only commands.
- The ranges come from the paper's hardware tests: 0.3 m/s forward, 0.15 m/s sideways,
  0.45 rad/s turning.

**Gait clock `gait`** (paper Sec. III-B, "phase input"):

- **Clock:** runs at **1 Hz** and starts at a random phase each episode.
- **Per-foot phase:** offsets for a **trot**: lf = 0, rf = π, lh = π, rh = 0, so the
  diagonal pairs move together.
- **Stance and swing:** φ ∈ [−π, 0) is stance and φ ∈ [0, π) is swing.
- **Purpose:** the phase is only an input to the policy and the foot-height reward. It does
  **not** generate joint commands; the policy produces all joint motion itself.

### Reward terms

`total = Σ weight · term · dt`, where dt = 0.02 s. mjlab applies the dt scaling, which
matches the paper's "w·dt".

| Term | Weight | Formula | Notes |
|---|---|---|---|
| `track_lin_vel` | 1.0 | exp(−‖v_cmd,xy − v_CoM,xy‖² / 0.05) | paper r_v |
| `track_ang_vel` | 0.5 | exp(−(ω_cmd,z − ω_CoM,z)² / 0.1) | paper r_ω |
| `energy` | −1e-3 | Σ \|q̇ᵢ · τᵢ\| | paper r_energy |
| `feet_phase` | 1.0 | exp(−Σⱼ (zⱼ − z_ref(φⱼ))² / 0.01) | paper r_phase; swing height 3 cm |
| `dof_pos_limits` | −1.0 | penalty for leaving the soft joint limits | **not in the paper** |
| `action_rate_l2` | −0.01 | ‖aₜ − aₜ₋₁‖² | **not in the paper** |

- **Why CoM velocity?** The body undulates, so the front segment (`base_link`, where the IMU
  sits) swings left and right with every stride. Tracking uses the **mass-weighted CoM
  velocity** instead (paper Sec. III-C). It is measured in a frame aligned with the line
  from the hind girdle (`body3`) to the front girdle (`base_link`).
- **The foot-height reference** z_ref(φ) is 0 during stance. During swing it rises and falls
  along two cubic Bézier halves (3s² − 2s³), peaking at 3 cm at mid-swing. With a zero
  command the reference is 0 for all feet, so the robot stands still instead of stepping
  in place.
- **σ values** (0.05, 0.1, 0.01) and the 3 cm swing height are **our choices**; the paper
  doesn't publish them.
- **The two extra penalties** guard against joint-limit abuse and jerky actions. Set their
  weight to 0 for the pure paper reward.

### Domain randomization (paper Sec. III-E)

| What | Range | When |
|---|---|---|
| Push (set base velocity) | vx, vy ±0.2 m/s, ωz ±0.4 rad/s | every 2–5 s |
| Foot friction | 0.4 – 1.2 | startup |
| Mass and inertia of all bodies | ±10% (`pseudo_inertia`, physically consistent) | startup |
| Spine body CoM offset | x, y ±1 cm, z ±0.5 cm | startup |
| Joint friction loss | 0.02 – 0.1 N·m | startup |
| Motor kp, kd | ×0.8 – 1.2 | every reset |
| Encoder bias | ±0.01 rad | startup |
| Spawn | x, y ±0.5 m, any yaw; joints ±0.05 rad | every reset |

Foot friction randomization only has an effect because foot geoms have contact `priority=1`,
which makes their friction override the floor's.

### PPO (paper Table II)

| | Paper | Here |
|---|---|---|
| Networks | MLP (512, 256, 128) | same, ELU |
| γ / learning rate / entropy | 0.97 / 3e-4 / 0.01 | same (fixed learning rate) |
| Rollout length | 20 | 20 |
| Epochs per batch | 4 | 4 |
| Gradient clipping | 1.0 | 1.0 |
| Observation normalization | yes | yes (included in the exported ONNX) |
| Envs × steps | 8192 × 3e8 | 4096 × 1500 iterations ≈ 1.2e8 was enough |

---

## 5. Train, watch, evaluate

```bash
# Train (about 23 min on a laptop RTX GPU with 4096 envs, about 89k steps/s)
uv run train Gecko-Locomotion-Flat --env.scene.num-envs 4096

# Watch the learning curves
uv run tensorboard --logdir logs/rsl_rl

# Watch a trained policy in the viewer (play needs an explicit checkpoint when not using W&B)
uv run play Gecko-Locomotion-Flat \
  --checkpoint-file logs/rsl_rl/gecko_locomotion/<run>/model_1499.pt --num-envs 4

# Drive the robot with the keyboard (latest checkpoint by default)
uv run python scripts/teleop.py [path/to/model_*.pt]

# Measure command tracking headlessly (achieved CoM velocity per command)
uv run python scripts/evaluate_policy.py logs/rsl_rl/gecko_locomotion/<run>/model_1499.pt
```

### Teleop keys

Focus the viewer window, then:

| Key | Effect |
|---|---|
| ↑ / ↓ | forward / backward (vx ± 0.05 m/s) |
| ← / → | turn left / right (ωz ± 0.1 rad/s) |
| `+` / `-` (also PageUp / PageDown, Home / End) | step left / right (vy ± 0.05 m/s) |
| Numpad 8/2, 4/6, 7/9, 5 | forward/back, step left/right, turn left/right, stop (Num Lock on) |
| Delete | stop (zero command) |
| Enter / Space | reset robot / pause |
| `[` / `]` | slow down / speed up the simulation |
| R | toggle the command and actual-velocity arrows |

The current command appears in the top-left corner. To change keys, edit `KEYMAP` at the
top of `scripts/teleop.py`; avoid letter keys, because MuJoCo uses every letter as a display
toggle. Commands are clamped to the training
ranges, because the policy has never seen anything outside them.

Each run writes to `logs/rsl_rl/gecko_locomotion/<date_time>/`:

- `model_<iter>.pt` checkpoints, saved every 100 iterations
- `<run>.onnx`, the deployable policy (see section 7)
- `params/`, the exact env and agent config used for that run

### Reference result

Run `2026-10-12_15-33-58`, `model_1499.pt`: 64 robots per command, measured over 6 s after a
2 s warm-up, with no pushes and no observation noise:

| Command (vx, vy, ωz) | Achieved CoM (vx, vy, ωz) | Falls | Power | Swing clearance |
|---|---|---|---|---|
| (0, 0, 0) | (0.000, 0.000, 0.000) | 0 | 0.8 W | 0.1 cm |
| (0.1, 0, 0) | (0.093, 0.001, 0.009) | 0 | 4.9 W | 2.6 cm |
| (0.2, 0, 0) | (0.196, 0.001, 0.015) | 0 | 6.8 W | 3.0 cm |
| (0.3, 0, 0) | (0.299, −0.001, 0.012) | 0 | 7.5 W | 3.1 cm |
| (−0.1, 0, 0) | (−0.096, 0.000, 0.003) | 0 | 3.8 W | 2.8 cm |
| (0, 0.1, 0) | (0.003, 0.094, 0.017) | 0 | 6.9 W | 2.9 cm |
| (0, 0, 0.4) | (−0.001, 0.001, 0.399) | 0 | 5.5 W | 2.6 cm |
| (0.2, 0, 0.3) | (0.197, 0.001, 0.297) | 0 | 7.6 W | 3.0 cm |

These numbers are from simulation only. They say nothing about whether the gait looks natural
or will transfer to hardware; watch it with `play` (section 5).

### Reading TensorBoard

- **Good signs:** `Episode_Reward/track_lin_vel` close to 1.0, `track_ang_vel` close to 0.5,
  `feet_phase` close to 1.0, mean episode length 1000, and `Episode_Termination/fell_over`
  close to 0. The reference run reached about 0.97 / 0.46 / 0.98 by iteration 600.
- **Misleading metric:** ignore `Metrics/twist/error_vel_yaw`. mjlab computes it at
  `base_link`, which swings with the body undulation, so it stays around 1.4 rad/s even when
  turning is tracked well. Use the rewards above or `evaluate_policy.py` instead.

---

## 6. Main things to know

1. **Motor parameters are placeholders.** `armature = 0.01` and `frictionloss = 0.05` are
   guesses. Before trying the real robot, system-ID one servo:
   - Mount it with a pendulum of known mass and length.
   - Log steps and chirps (goal, present position, velocity, current).
   - Fit armature, friction, kp scale and latency so the sim reproduces the logs.

   The paper credits this kind of servo matching for its sim-to-real success.
2. **Command latency is not modelled yet.** Measure your loop latency (bus plus Raspberry Pi),
   then set `delay_min_lag` / `delay_max_lag` on `GECKO_ACTUATOR_CFG`. The units are physics
   steps of 5 ms.
3. **Observation noise is generic.** The paper fits noise ranges to real sensor logs. Record
   the IMU and joint readings on the robot and update the `Unoise` ranges.
4. **No backlash.** If real foot clearance comes out lower than in sim, gear backlash is the
   first suspect. The paper fixed this with ±4° passive joints at the girdles.
5. **Edit the source, not the generated files.** After changing the USD, re-run the converter
   (section 1); never edit `gecko.xml` by hand. The `FOOT_COLLISION_AS_CYLINDER` and
   `MAX_HULL_VERT` options are at the top of the converter.
6. **The nominal pose is the CAD pose.** If you change `INIT_STATE.joint_pos`, also update
   `pos` z, which is the base height. Then re-run `pytest tests`: it checks that the robot
   stands still with only its feet touching the floor.
7. **Gait settings are free parameters.** Frequency, trot offsets and swing height live in
   `commands["gait"]` and `rewards["feet_phase"]`. A slower gait (0.6–0.8 Hz) or a lateral
   sequence walk may suit the gecko better than the default trot.

---

## 7. Deploying the ONNX policy

`<run>.onnx` takes `obs` of shape [1, 112] and returns `actions` of shape [1, 19].
Observation normalization is built into the model. The file's metadata includes the joint
names, default joint positions and action scales.

Every 20 ms on the robot:

1. **Read sensors:** the joint positions and velocities of the 19 actuated joints, in the
   **joint order of section 4**, plus the IMU gyro and gravity direction in the front-girdle
   (`base_link`) frame.
2. **Update the gait clock:** φ_clock += 2π · 1 Hz · 0.02 s. Each foot's phase is
   φ_clock + offset, with offsets lf = 0, rf = π, lh = π, rh = 0, wrapped to [−π, π).
3. **Build the observation:** the 112 dims in the order of the actor table. Joint positions
   are relative to the nominal pose (0 rad). Leave out the passive foot joints.
4. **Run the policy:** feed the observation to the ONNX model to get `a`, and keep the last
   3 actions for the next observation, oldest first.
5. **Compute targets:** `x = 0 + scale · a`, using a scale of 0.25 for the spine and 0.5 for
   the legs. Then filter: `y += 0.40 · (x − y)`.
6. **Send** `y` as goal positions in radians, converted to Dynamixel ticks for your
   zero offsets.

The ONNX metadata `joint_names` lists all 27 joints, including the 8 passive feet. The policy
itself only uses the 19 actuated joints, in the order of section 4.
