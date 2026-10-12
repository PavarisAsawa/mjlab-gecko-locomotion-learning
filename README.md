# mj_lab_gecko_project

RL training for a gecko robot in simulation with [mjlab](https://github.com/mujocolab/mjlab).

## Setup

```bash
uv sync                          # Python 3.13 venv (mjlab requires <3.14)
export GIT_PYTHON_REFRESH=quiet  # only needed if git is not installed
```

## Usage

```bash
uv run list-envs                                   # Gecko-Locomotion-Flat
uv run train Gecko-Locomotion-Flat --env.scene.num-envs 4096
uv run play  Gecko-Locomotion-Flat --checkpoint-file logs/rsl_rl/gecko_locomotion/<run>/model_1499.pt
uv run tensorboard --logdir logs/rsl_rl
uv run python tests/spawn_gecko.py                 # robot standing on the floor
uv run pytest tests

# After changing the USD, regenerate the MJCF:
uv run --no-project --python 3.12 --with usd-core --with numpy python scripts/usd_to_mjcf.py \
  src/mj_lab_gecko_project/robots/gecko/gecko-aug-2026.usd src/mj_lab_gecko_project/robots/gecko/xmls
```

## Layout

```
src/mj_lab_gecko_project/
├── robots/gecko/              # USD source, generated MJCF, EntityCfg (XM430 actuators)
└── tasks/locomotion/
    ├── locomotion_env_cfg.py  # flat-ground task (obs, actions, rewards, DR)
    ├── mdp/                   # gait clock, filtered action, CoM velocity, rewards
    └── config/gecko/          # gecko env cfg, PPO cfg, task registration
```

Tasks are registered through the `mjlab.tasks` entry point in `pyproject.toml`,
so mjlab's `train` / `play` / `list-envs` find them automatically. Re-run
`uv sync` after changing entry points.

## Task design

Full write-up: [docs/LOCOMOTION.md](docs/LOCOMOTION.md).

`Gecko-Locomotion-Flat` follows the terrestrial part of Tian et al. 2026,
*Learning Whole-Body Control for a Salamander Robot* (arXiv:2603.16683), on
flat ground and without backlash modelling:

- **Action**: residual joint offsets on the nominal pose (19 joints), 4 Hz
  first-order low-pass, 50 Hz policy.
- **Actor obs**: joint vel, joint offset, IMU angular velocity, projected
  gravity, gait phase (sin/cos per foot), velocity command, last 3 actions.
  No linear velocity. The critic also gets CoM velocity and foot heights.
- **Rewards** (x dt): CoM linear velocity tracking (1.0), CoM yaw-rate tracking
  (0.5), energy |q̇·τ| (-1e-3), phase-conditioned Bezier foot height (1.0), plus
  joint-limit (-1.0) and action-rate (-0.01) regularizers not in the paper.
- **DR**: pushes, foot friction, mass/COM, joint friction, motor gains, encoder bias.
