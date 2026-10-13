# mj_lab_gecko_project

RL training for a gecko robot in simulation with [mjlab](https://github.com/mujocolab/mjlab).

## Setup

```bash
uv sync                          # Python 3.13 venv (mjlab requires <3.14)
export GIT_PYTHON_REFRESH=quiet  # only needed if git is not installed
```

## Usage

```bash
uv run list-envs                                   # Gecko-Locomotion-Flat / -Rough
uv run train Gecko-Locomotion-Flat --env.scene.num-envs 4096
uv run play  Gecko-Locomotion-Flat
uv run tensorboard --logdir logs/rsl_rl
uv run python scripts/view_robot.py                # inspect the robot model
uv run python scripts/spawn_gecko.py                # inspect the robot model and testing on floor
uv run pytest tests
```

## Layout

```
src/mj_lab_gecko_project/
├── robots/gecko/              # MJCF + EntityCfg (actuators, init state, collisions)
└── tasks/locomotion/
    ├── locomotion_env_cfg.py  # robot-agnostic base env cfg
    ├── mdp/                   # custom rewards/observations/terminations/curriculums
    └── config/gecko/          # gecko env cfgs, PPO cfg, task registration
```

Tasks are registered through the `mjlab.tasks` entry point in `pyproject.toml`,
so mjlab's `train` / `play` / `list-envs` find them automatically. Re-run
`uv sync` after changing entry points.

**Note:** `robots/gecko/xmls/gecko.xml` is currently a copy of the Unitree Go1
model as a placeholder. Replace it with the gecko model and update
`gecko_constants.py` and `config/gecko/env_cfgs.py` (body/site/geom names).

## Docs

- [docs/LOCOMOTION.md](docs/LOCOMOTION.md): robot, actuators, task design, training, deployment
- [docs/COMPARISON.md](docs/COMPARISON.md): MLP/Hebbian × PPO/OpenES comparison, with videos
